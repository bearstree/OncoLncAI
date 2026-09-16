"""Versioned, leakage-safe evaluation for structured evidence extraction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oncolncai.evidence import (
    AssociationDirection,
    BiomarkerRole,
    EvidenceExtractionRequest,
    EvidenceRecord,
    EvidenceType,
    ExpressionDirection,
    LiteratureSource,
    extract_literature_evidence,
)
from oncolncai.providers import LLMProvider


class GoldEvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    lncrna: str = Field(min_length=1)
    cancer: str = Field(min_length=1)
    evidence_type: EvidenceType | None = None
    biomarker_role: BiomarkerRole | None = None
    expression_direction: ExpressionDirection | None = None
    association_direction: AssociationDirection | None = None
    experimental_validation: bool | None = None


class ExtractionBenchmarkExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    example_id: str = Field(pattern=r"^[a-z0-9_]+$")
    split: str = Field(pattern=r"^(train|validation|test)$")
    source: LiteratureSource
    gold_records: tuple[GoldEvidenceRecord, ...] = ()
    review_status: str = Field(pattern=r"^(manually_reviewed|deterministic_silver)$")


class ExtractionBenchmarkDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = Field(pattern=r"^1\.0\.0$")
    dataset_id: str = Field(min_length=1)
    examples: tuple[ExtractionBenchmarkExample, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_leakage_and_coverage(self) -> ExtractionBenchmarkDataset:
        ids = [item.example_id for item in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("example IDs must be unique")
        pmid_splits: dict[str, set[str]] = {}
        for item in self.examples:
            pmid_splits.setdefault(item.source.pmid, set()).add(item.split)
        if any(len(splits) > 1 for splits in pmid_splits.values()):
            raise ValueError("a PMID may occur in only one split")
        if {item.split for item in self.examples} != {"train", "validation", "test"}:
            raise ValueError("train, validation, and test splits are all required")
        if not any(item.gold_records for item in self.examples if item.split == "test"):
            raise ValueError("test split must include positive gold evidence")
        if any(item.review_status != "manually_reviewed" for item in self.examples if item.split == "test"):
            raise ValueError("every frozen test example must be manually reviewed")
        return self


class ExtractionCorpusReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    ready_for_pilot_lora: bool
    train_examples: int
    validation_examples: int
    test_examples: int
    unique_pmids: int
    cancers: tuple[str, ...]
    manually_reviewed_test_examples: int
    minimum_train_examples: int = 120
    minimum_validation_examples: int = 20
    required_cancers: tuple[str, ...] = ("lung adenocarcinoma", "breast cancer", "clear cell renal cell carcinoma")
    limitations: tuple[str, ...]


def assess_corpus_readiness(dataset: ExtractionBenchmarkDataset) -> ExtractionCorpusReadiness:
    counts = {split: sum(item.split == split for item in dataset.examples) for split in ("train", "validation", "test")}
    cancers = tuple(sorted({item.source.cancer.casefold() for item in dataset.examples}))
    reviewed_test = sum(item.split == "test" and item.review_status == "manually_reviewed" for item in dataset.examples)
    required = {"lung adenocarcinoma", "breast cancer", "clear cell renal cell carcinoma"}
    ready = counts["train"] >= 120 and counts["validation"] >= 20 and required.issubset(cancers) and reviewed_test == counts["test"] and counts["test"] >= 4
    limitations = (
        "Silver train/validation labels cover entity and cancer grounding; richer evidence fields need expert annotation.",
        "The four-example frozen gold test set is adequate for regression but too small for a robust performance claim.",
    )
    return ExtractionCorpusReadiness(
        dataset_id=dataset.dataset_id, ready_for_pilot_lora=ready,
        train_examples=counts["train"], validation_examples=counts["validation"], test_examples=counts["test"],
        unique_pmids=len({item.source.pmid for item in dataset.examples}), cancers=cancers,
        manually_reviewed_test_examples=reviewed_test, limitations=limitations,
    )


class ExtractionPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    example_id: str
    records: tuple[EvidenceRecord, ...] = ()
    error: str | None = None


class ExtractionBaselineRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    schema_version: str
    provider: str
    model: str
    extraction_interface: str = "extract_literature_evidence"
    created_at: datetime
    predictions: tuple[ExtractionPrediction, ...]


class ExtractionFailureCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    example_id: str
    false_positive_atoms: tuple[str, ...] = ()
    false_negative_atoms: tuple[str, ...] = ()
    error: str | None = None


class ExtractionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    split: str
    example_count: int = Field(ge=1)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    failure_cases: tuple[ExtractionFailureCase, ...]


def load_extraction_dataset(path: Path) -> ExtractionBenchmarkDataset:
    return ExtractionBenchmarkDataset.model_validate_json(path.read_text(encoding="utf-8"))


def _normal(value: str) -> str:
    aliases = {"luad": "lung adenocarcinoma", "nsclc": "non small cell lung cancer"}
    normalized = " ".join(value.casefold().replace("-", " ").split())
    return aliases.get(normalized, normalized)


def _atoms(record: GoldEvidenceRecord | EvidenceRecord) -> set[str]:
    prefix = f"{_normal(record.lncrna)}|{_normal(record.cancer)}"
    atoms = {f"{prefix}|record"}
    for field in ("evidence_type", "biomarker_role", "expression_direction", "association_direction", "experimental_validation"):
        value = getattr(record, field)
        if value is not None:
            rendered = value.value if hasattr(value, "value") else str(value).casefold()
            atoms.add(f"{prefix}|{field}={rendered}")
    return atoms


def evaluate_extraction_baseline(dataset: ExtractionBenchmarkDataset, run: ExtractionBaselineRun, *, split: str = "test") -> ExtractionMetrics:
    if run.dataset_id != dataset.dataset_id or run.schema_version != dataset.schema_version:
        raise ValueError("baseline run does not match benchmark dataset")
    predictions = {item.example_id: item for item in run.predictions}
    if len(predictions) != len(run.predictions):
        raise ValueError("baseline prediction example IDs must be unique")
    selected = tuple(item for item in dataset.examples if item.split == split)
    if not {item.example_id for item in selected}.issubset(predictions):
        raise ValueError("baseline predictions must cover every selected benchmark example")
    tp = fp = fn = 0
    failures: list[ExtractionFailureCase] = []
    for example in selected:
        prediction = predictions[example.example_id]
        gold_atoms = set().union(*(_atoms(item) for item in example.gold_records)) if example.gold_records else set()
        predicted_atoms = set().union(*(_atoms(item) for item in prediction.records)) if prediction.records else set()
        false_positive = tuple(sorted(predicted_atoms - gold_atoms))
        false_negative = tuple(sorted(gold_atoms - predicted_atoms))
        tp += len(gold_atoms & predicted_atoms)
        fp += len(false_positive)
        fn += len(false_negative)
        if false_positive or false_negative or prediction.error:
            failures.append(ExtractionFailureCase(example_id=example.example_id, false_positive_atoms=false_positive, false_negative_atoms=false_negative, error=prediction.error))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ExtractionMetrics(split=split, example_count=len(selected), true_positives=tp, false_positives=fp, false_negatives=fn, precision=precision, recall=recall, f1=f1, failure_cases=tuple(failures))


def run_extraction_baseline(dataset: ExtractionBenchmarkDataset, *, provider: LLMProvider, provider_name: str, model: str, split: str | None = None) -> ExtractionBaselineRun:
    predictions: list[ExtractionPrediction] = []
    examples = dataset.examples if split is None else tuple(item for item in dataset.examples if item.split == split)
    for example in examples:
        result = extract_literature_evidence(EvidenceExtractionRequest(source=example.source), provider=provider)
        if result.data is None:
            predictions.append(ExtractionPrediction(example_id=example.example_id, error=result.error.message if result.error else "unknown extraction failure"))
        else:
            predictions.append(ExtractionPrediction(example_id=example.example_id, records=result.data.records))
    return ExtractionBaselineRun(dataset_id=dataset.dataset_id, schema_version=dataset.schema_version, provider=provider_name, model=model, created_at=datetime.now(timezone.utc), predictions=tuple(predictions))


def save_baseline_artifacts(run: ExtractionBaselineRun, metrics: ExtractionMetrics, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "baseline_predictions.json").write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (directory / "baseline_metrics.json").write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")
