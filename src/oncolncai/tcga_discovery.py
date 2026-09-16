"""Real genome-wide TCGA preparation and deterministic candidate discovery."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oncolncai.differential_expression import (
    CountFeature,
    DifferentialExpressionMethod,
    DifferentialExpressionRequest,
    DifferentialExpressionResult,
    ExpressionDataType,
    ExpressionGroup,
    ExpressionSample,
    run_differential_expression,
)
from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult
from oncolncai.tcga_survival import UrllibDownloadTransport, gdc_expression_url


GENCODE_V36_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_36/"
    "gencode.v36.annotation.gtf.gz"
)

LNCRNA_BIOTYPES = frozenset(
    {
        "3prime_overlapping_ncRNA", "antisense", "bidirectional_promoter_lncRNA",
        "lincRNA", "lncRNA", "macro_lncRNA", "non_coding", "processed_transcript",
        "sense_intronic", "sense_overlapping", "TEC",
    }
)


class ResearchMode(StrEnum):
    GENOME_WIDE_DISCOVERY = "GENOME_WIDE_DISCOVERY"
    SINGLE_CANDIDATE_VALIDATION = "SINGLE_CANDIDATE_VALIDATION"
    MULTI_CANDIDATE_VALIDATION = "MULTI_CANDIDATE_VALIDATION"


class CapabilityStatus(StrEnum):
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    INTEGRATION_INCOMPLETE = "INTEGRATION_INCOMPLETE"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    FAILED = "FAILED"
    SKIPPED_AFTER_REPLAN = "SKIPPED_AFTER_REPLAN"


class AnalysisCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis: str
    implementation: str = "available"
    data_available: bool
    inputs_valid: bool
    eligible: bool
    status: CapabilityStatus
    reason: str | None = None


class GeneAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ensembl_id: str
    normalized_ensembl_id: str = Field(pattern=r"^ENSG\d{11}$")
    gene_symbol: str
    gene_biotype: str


class SampleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    case_id: str
    patient_id: str
    sample_type: str
    tumor_normal_status: ExpressionGroup


class ClinicalMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    patient_id: str
    overall_survival_time: float = Field(gt=0)
    event_status: int = Field(ge=0, le=1)
    age: float | None = Field(default=None, gt=0)
    sex: str | None = None
    pathological_stage: str | None = None


class GenomeWidePreparationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_columns: int = Field(ge=0)
    selected_samples: int = Field(ge=0)
    tumor_samples: int = Field(ge=0)
    normal_samples: int = Field(ge=0)
    duplicate_aliquots_removed: int = Field(ge=0)
    duplicate_gene_rows_removed: int = Field(default=0, ge=0)
    genes_in_source: int = Field(ge=0)
    annotated_genes: int = Field(ge=0)
    lncrnas_retained: int = Field(ge=0)


class PreparedGenomeWideCohort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project_id: str
    samples: tuple[SampleMetadata, ...]
    features: tuple[CountFeature, ...]
    annotations: tuple[GeneAnnotation, ...]
    clinical: tuple[ClinicalMetadata, ...] = ()
    preparation: GenomeWidePreparationSummary
    expression_url: str
    annotation_url: str
    expression_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: datetime
    expression_preprocessing: str = "inverse log2(count+1), rounded to non-negative integer STAR count"
    expression_data_type: ExpressionDataType = ExpressionDataType.XENA_LOG2_STAR_COUNTS
    expression_transform: str = "UCSC Xena averages duplicate sample-vial counts then applies log2(count + 1)"
    expression_source_dataset: str = "TCGA project STAR - Counts matrix"
    upstream_source: str = "NCI GDC / TCGA; STAR - Counts"
    distribution_source: str = "UCSC Xena GDC Hub"
    expression_metadata_url: str | None = None
    expression_dataset_version: str | None = None
    upstream_gdc_release: str | None = None

    def de_request(self, *, min_cases: int = 20, min_controls: int = 10) -> DifferentialExpressionRequest:
        return DifferentialExpressionRequest(
            project_id=self.project_id,
            samples=tuple(ExpressionSample(sample_id=s.sample_id, group=s.tumor_normal_status) for s in self.samples),
            features=self.features,
            min_case_samples=min_cases,
            min_control_samples=min_controls,
            count_source=f"UCSC Xena GDC-hub STAR counts sha256:{self.expression_sha256}",
            annotation_source=f"GENCODE v36 sha256:{self.annotation_sha256}",
            method=DifferentialExpressionMethod.AUTO,
            input_data_type=self.expression_data_type,
            expression_transform=self.expression_transform,
            expression_source_dataset=self.expression_source_dataset,
            upstream_source=self.upstream_source,
            distribution_source=self.distribution_source,
        )


class CandidateSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fdr_threshold: float = Field(default=0.05, gt=0, le=1)
    absolute_log2_fc_threshold: float = Field(default=1.0, ge=0)
    minimum_mean_cpm: float = Field(default=1.0, ge=0)
    top_n: int = Field(default=10, ge=1, le=100)


class DiscoveredCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ensembl_gene_id: str
    gene_symbol: str
    gene_biotype: str
    log2_fold_change: float
    p_value: float
    fdr: float
    maximum_mean_cpm: float
    selection_rule: str


class CandidateDiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidates: tuple[DiscoveredCandidate, ...]
    configuration: CandidateSelectionConfig
    eligible_result_count: int = Field(ge=0)
    qualified_feature_ids: tuple[str, ...] = ()


class GenomeWideDataRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project_id: str = Field(pattern=r"^TCGA-[A-Z0-9]+$")
    expression_url: str | None = None
    annotation_url: str = GENCODE_V36_GTF_URL


def xena_expression_metadata_url(project_id: str) -> str:
    return f"https://gdc.xenahubs.net/download/{project_id}.star_counts.tsv.json"


class DownloadTransport(Protocol):
    def download(self, url: str, destination: Path, timeout: float) -> None: ...


class TCGAGenomeWideClient:
    def __init__(self, *, transport: DownloadTransport | None = None,
                 cache_dir: Path = Path(".cache/oncolncai/tcga_discovery"),
                 timeout: float = 300.0,
                 clock: Callable[[], datetime] | None = None) -> None:
        self._transport = transport or UrllibDownloadTransport()
        self._cache_dir = cache_dir
        self._timeout = timeout
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def prepare(self, request: GenomeWideDataRequest) -> ToolResult[PreparedGenomeWideCohort]:
        expression_url = request.expression_url or gdc_expression_url(request.project_id)
        try:
            expression_path = self._obtain(expression_url)
            annotation_path = self._obtain(request.annotation_url)
            metadata_url = xena_expression_metadata_url(request.project_id)
            metadata_path = self._obtain(metadata_url)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("unit") != "log2(count+1)" or metadata.get("label") != "STAR - Counts":
                raise ValueError("Xena expression metadata did not declare the expected STAR - Counts log2(count+1) representation")
            cohort = prepare_genome_wide_cohort(
                request.project_id, expression_path=expression_path,
                annotation_path=annotation_path, expression_url=expression_url,
                annotation_url=request.annotation_url, retrieved_at=self._clock(),
                expression_metadata_url=metadata_url,
                expression_dataset_version=metadata.get("version"),
                upstream_gdc_release=metadata.get("url"),
            )
            return ToolResult(status=ResultStatus.SUCCESS, data=cohort, provenance=[
                Provenance(source="UCSC Xena GDC Hub", source_id=expression_url,
                           dataset_version=f"sha256:{cohort.expression_sha256}"),
                Provenance(source="GENCODE", source_id=request.annotation_url,
                           dataset_version=f"v36;sha256:{cohort.annotation_sha256}"),
            ])
        except (OSError, ValueError, KeyError, UnicodeError) as exc:
            return ToolResult(status=ResultStatus.FAILURE,
                              error=ToolError(code="tcga_genome_wide_preparation_failed", message=str(exc), retryable=False),
                              provenance=[Provenance(source="UCSC Xena GDC Hub", source_id=expression_url),
                                          Provenance(source="GENCODE", source_id=request.annotation_url)])

    def _obtain(self, url: str) -> Path:
        suffix = ".gz" if url.endswith(".gz") else ".json" if url.endswith(".json") else ".tsv"
        path = self._cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}{suffix}"
        if not path.exists() or not path.stat().st_size:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            self._transport.download(url, temporary, self._timeout)
            if not temporary.stat().st_size:
                raise ValueError(f"downloaded source is empty: {url}")
            temporary.replace(path)
        return path


def normalize_ensembl_id(value: str) -> str:
    return value.split(".", 1)[0]


def classify_tcga_sample(sample_id: str) -> ExpressionGroup | None:
    parts = sample_id.split("-")
    if len(parts) < 4:
        raise ValueError(f"invalid TCGA sample barcode: {sample_id}")
    code = parts[3][:2]
    return ExpressionGroup.CASE if code == "01" else ExpressionGroup.CONTROL if code == "11" else None


def _patient_id(sample_id: str) -> str:
    return "-".join(sample_id.split("-")[:3])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _gtf_attributes(text: str) -> dict[str, str]:
    return {key: value for key, value in re.findall(r'(\w+) "([^"]+)"', text)}


def read_gencode_lncrnas(path: Path) -> dict[str, GeneAnnotation]:
    opener = gzip.open if path.suffix == ".gz" else open
    annotations: dict[str, GeneAnnotation] = {}
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            values = _gtf_attributes(fields[8])
            biotype = values.get("gene_type") or values.get("gene_biotype")
            identifier = values.get("gene_id")
            symbol = values.get("gene_name")
            if not identifier or not symbol or biotype not in LNCRNA_BIOTYPES:
                continue
            normalized = normalize_ensembl_id(identifier)
            annotations[normalized] = GeneAnnotation(
                ensembl_id=identifier, normalized_ensembl_id=normalized,
                gene_symbol=symbol, gene_biotype=biotype,
            )
    if not annotations:
        raise ValueError("GENCODE annotation contained no recognized lncRNA genes")
    return annotations


def prepare_genome_wide_cohort(project_id: str, *, expression_path: Path,
                               annotation_path: Path, expression_url: str,
                               annotation_url: str,
                               expression_metadata_url: str | None = None,
                               expression_dataset_version: str | None = None,
                               upstream_gdc_release: str | None = None,
                               retrieved_at: datetime | None = None) -> PreparedGenomeWideCohort:
    annotation = read_gencode_lncrnas(annotation_path)
    opener = gzip.open if expression_path.suffix == ".gz" else open
    with opener(expression_path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        header = next(reader)
        source_samples = header[1:]
        selected_by_key: dict[tuple[str, ExpressionGroup], tuple[str, int]] = {}
        for index, sample_id in enumerate(source_samples):
            group = classify_tcga_sample(sample_id)
            if group is None:
                continue
            key = (_patient_id(sample_id), group)
            current = selected_by_key.get(key)
            if current is None or sample_id < current[0]:
                selected_by_key[key] = (sample_id, index)
        selected = sorted(selected_by_key.values())
        samples = tuple(SampleMetadata(
            sample_id=sample_id, case_id=_patient_id(sample_id), patient_id=_patient_id(sample_id),
            sample_type="Primary Tumor" if classify_tcga_sample(sample_id) == ExpressionGroup.CASE else "Solid Tissue Normal",
            tumor_normal_status=classify_tcga_sample(sample_id),
        ) for sample_id, _ in selected)
        features: list[CountFeature] = []
        seen_features: set[str] = set()
        duplicate_gene_rows = 0
        genes = 0
        for row in reader:
            if not row:
                continue
            genes += 1
            normalized = normalize_ensembl_id(row[0])
            gene = annotation.get(normalized)
            if gene is None:
                continue
            if normalized in seen_features:
                duplicate_gene_rows += 1
                continue
            if len(row) - 1 != len(source_samples):
                raise ValueError(f"expression row width mismatch for {row[0]}")
            counts = tuple(max(0, round(math.pow(2.0, float(row[index + 1])) - 1.0)) for _, index in selected)
            features.append(CountFeature(feature_id=normalized, gene_symbol=gene.gene_symbol, counts=counts))
            seen_features.add(normalized)
    if not samples or not features:
        raise ValueError("no eligible samples or annotated lncRNAs were prepared")
    tumors = sum(s.tumor_normal_status == ExpressionGroup.CASE for s in samples)
    normals = len(samples) - tumors
    return PreparedGenomeWideCohort(
        project_id=project_id, samples=samples, features=tuple(features),
        annotations=tuple(annotation[f.feature_id] for f in features),
        preparation=GenomeWidePreparationSummary(
            source_columns=len(source_samples), selected_samples=len(samples), tumor_samples=tumors,
            normal_samples=normals, duplicate_aliquots_removed=sum(classify_tcga_sample(s) is not None for s in source_samples) - len(samples),
            duplicate_gene_rows_removed=duplicate_gene_rows,
            genes_in_source=genes, annotated_genes=len(features), lncrnas_retained=len(features),
        ), expression_url=expression_url, annotation_url=annotation_url,
        expression_sha256=_sha256(expression_path), annotation_sha256=_sha256(annotation_path),
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
        expression_metadata_url=expression_metadata_url,
        expression_dataset_version=expression_dataset_version,
        upstream_gdc_release=upstream_gdc_release,
    )


def discover_candidates(result: DifferentialExpressionResult, annotations: tuple[GeneAnnotation, ...],
                        config: CandidateSelectionConfig = CandidateSelectionConfig()) -> CandidateDiscoveryResult:
    by_id = {item.normalized_ensembl_id: item for item in annotations}
    eligible = [item for item in result.results if item.adjusted_p_value <= config.fdr_threshold
                and abs(item.log2_fold_change) >= config.absolute_log2_fc_threshold
                and max(item.case_mean_cpm, item.control_mean_cpm) >= config.minimum_mean_cpm]
    eligible.sort(key=lambda item: (item.adjusted_p_value, -abs(item.log2_fold_change), item.feature_id))
    rule = (f"FDR <= {config.fdr_threshold}; abs(log2FC) >= {config.absolute_log2_fc_threshold}; "
            f"max mean CPM >= {config.minimum_mean_cpm}; top {config.top_n}")
    candidates = tuple(DiscoveredCandidate(
        ensembl_gene_id=item.feature_id, gene_symbol=item.gene_symbol or item.feature_id,
        gene_biotype=by_id[item.feature_id].gene_biotype,
        log2_fold_change=item.log2_fold_change, p_value=item.p_value, fdr=item.adjusted_p_value,
        maximum_mean_cpm=max(item.case_mean_cpm, item.control_mean_cpm), selection_rule=rule,
    ) for item in eligible[:config.top_n])
    return CandidateDiscoveryResult(
        candidates=candidates, configuration=config, eligible_result_count=len(eligible),
        qualified_feature_ids=tuple(item.feature_id for item in eligible),
    )


def run_real_genome_wide_de(cohort: PreparedGenomeWideCohort) -> ToolResult[DifferentialExpressionResult]:
    """Exact adapter call into the existing deterministic DE implementation."""
    return run_differential_expression(cohort.de_request())
