"""Structured, provenance-preserving extraction of literature evidence."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from oncolncai.providers import LLMProvider, MockLLMProvider
from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult


class EvidenceType(StrEnum):
    EXPRESSION = "expression"
    DIAGNOSTIC = "diagnostic"
    PROGNOSTIC = "prognostic"
    MECHANISTIC = "mechanistic"
    THERAPEUTIC = "therapeutic"
    OTHER = "other"


class BiomarkerRole(StrEnum):
    DIAGNOSTIC = "diagnostic"
    PROGNOSTIC = "prognostic"
    PREDICTIVE = "predictive"
    MONITORING = "monitoring"


class ExpressionDirection(StrEnum):
    UPREGULATED = "upregulated"
    DOWNREGULATED = "downregulated"
    MIXED = "mixed"
    UNCHANGED = "unchanged"


class AssociationDirection(StrEnum):
    FAVORABLE = "favorable"
    ADVERSE = "adverse"
    NONE = "none"
    MIXED = "mixed"


class LiteratureSource(BaseModel):
    """Retrieved source text and its immutable PubMed context."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    pmid: str = Field(pattern=r"^\d+$")
    cancer: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    query: str | None = None


class SourcePassage(BaseModel):
    """Bounded excerpt passed to the structured extraction provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_id: str = Field(min_length=1)
    pmid: str = Field(pattern=r"^\d+$")
    text: str = Field(min_length=1)


class EvidenceRecord(BaseModel):
    """A single source-grounded lncRNA claim."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    lncrna: str = Field(min_length=1)
    cancer: str = Field(min_length=1)
    pmid: str = Field(pattern=r"^\d+$")
    passage_id: str = Field(min_length=1)
    evidence_type: EvidenceType | None = None
    biomarker_role: BiomarkerRole | None = None
    expression_direction: ExpressionDirection | None = None
    outcome: str | None = None
    association_direction: AssociationDirection | None = None
    sample_type: str | None = None
    cohort_size: int | None = Field(default=None, gt=0)
    experimental_validation: bool | None = None
    supporting_text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class EvidenceBatch(BaseModel):
    """Schema requested from a structured-generation provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[EvidenceRecord, ...] = ()


class EvidenceExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: LiteratureSource
    max_chunk_chars: int = Field(default=2000, ge=200, le=8000)
    overlap_chars: int = Field(default=200, ge=0, le=1000)

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> EvidenceExtractionRequest:
        if self.overlap_chars >= self.max_chunk_chars:
            raise ValueError("overlap_chars must be smaller than max_chunk_chars")
        return self


class EvidenceExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[EvidenceRecord, ...] = ()
    passages_processed: int = Field(ge=0)
    provider_calls: int = Field(ge=0)


StructuredGenerationProvider = LLMProvider
MockStructuredProvider = MockLLMProvider


def chunk_source(source: LiteratureSource, max_chars: int, overlap: int) -> tuple[SourcePassage, ...]:
    """Split source text at whitespace into bounded, overlapping passages."""

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")

    text = source.text
    passages: list[SourcePassage] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end + 1)
            if boundary > start:
                end = boundary
        passage_text = text[start:end].strip()
        if passage_text:
            passages.append(
                SourcePassage(
                    passage_id=f"pmid:{source.pmid}:passage:{len(passages) + 1}",
                    pmid=source.pmid,
                    text=passage_text,
                )
            )
        if end >= len(text):
            break
        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start
    return tuple(passages)


def build_evidence_extraction_prompt(source: LiteratureSource, passage: SourcePassage) -> str:
    return (
        "Extract only lncRNA evidence explicitly supported by this passage. "
        "Use null for unreported fields and copy supporting_text from the passage. "
        "Confidence is semantic extraction confidence, not statistical significance: "
        "use 0.90-1.00 for an explicit unambiguous claim, 0.70-0.89 for supported "
        "evidence with minor ambiguity, 0.50-0.69 for indirect or qualified evidence, "
        "and below 0.50 for weak or unclear evidence.\n"
        f"Cancer: {source.cancer}\n"
        f"PMID: {source.pmid}\n"
        f"Passage ID: {passage.passage_id}\n"
        f"Passage:\n{passage.text}"
    )


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def normalize_lncrna_identifier(value: str) -> str:
    """Remove a generic lncRNA label without guessing biological aliases."""

    normalized = re.sub(r"^lncRNA[\s_:-]*", "", value.strip(), flags=re.IGNORECASE)
    return normalized or value.strip()


def _validate_grounding(
    record: EvidenceRecord,
    source: LiteratureSource,
    passage: SourcePassage,
) -> None:
    if record.pmid != source.pmid:
        raise ValueError(f"evidence PMID {record.pmid} does not match source PMID {source.pmid}")
    if record.cancer.casefold() != source.cancer.casefold():
        raise ValueError("evidence cancer does not match source cancer")
    if record.passage_id != passage.passage_id:
        raise ValueError("evidence passage_id does not match the source passage")
    if _normalized_text(record.supporting_text) not in _normalized_text(passage.text):
        raise ValueError("supporting_text is not present in the source passage")


def extract_literature_evidence(
    request: EvidenceExtractionRequest,
    *,
    provider: StructuredGenerationProvider,
) -> ToolResult[EvidenceExtractionResult]:
    """Extract and validate evidence from bounded passages using a supplied provider."""

    passages = chunk_source(
        request.source,
        request.max_chunk_chars,
        request.overlap_chars,
    )
    records: list[EvidenceRecord] = []
    try:
        for passage in passages:
            raw_output = provider.generate_structured(
                prompt=build_evidence_extraction_prompt(request.source, passage),
                output_schema=EvidenceBatch,
            )
            batch = EvidenceBatch.model_validate(raw_output)
            for record in batch.records:
                _validate_grounding(record, request.source, passage)
                records.append(
                    record.model_copy(
                        update={"lncrna": normalize_lncrna_identifier(record.lncrna)}
                    )
                )
    except (ValidationError, ValueError) as exc:
        return ToolResult[EvidenceExtractionResult](
            status=ResultStatus.FAILURE,
            error=ToolError(
                code="invalid_evidence_output",
                message=str(exc),
                retryable=False,
            ),
            provenance=[
                Provenance(
                    source="NCBI PubMed",
                    source_id=request.source.pmid,
                    query=request.source.query,
                )
            ],
        )
    except Exception as exc:
        return ToolResult[EvidenceExtractionResult](
            status=ResultStatus.FAILURE,
            error=ToolError(
                code="evidence_provider_failed",
                message=str(exc) or type(exc).__name__,
                retryable=True,
            ),
            provenance=[
                Provenance(
                    source="NCBI PubMed",
                    source_id=request.source.pmid,
                    query=request.source.query,
                )
            ],
        )

    unique_records: list[EvidenceRecord] = []
    seen: set[str] = set()
    for record in records:
        key = json.dumps(
            record.model_dump(mode="json", exclude={"passage_id"}),
            sort_keys=True,
        )
        if key not in seen:
            seen.add(key)
            unique_records.append(record)

    result = EvidenceExtractionResult(
        records=tuple(unique_records),
        passages_processed=len(passages),
        provider_calls=len(passages),
    )
    warnings = [] if unique_records else ["No supported lncRNA evidence was extracted."]
    return ToolResult[EvidenceExtractionResult](
        status=ResultStatus.SUCCESS,
        data=result,
        warnings=warnings,
        provenance=[
            Provenance(
                source="NCBI PubMed",
                source_id=request.source.pmid,
                query=request.source.query,
            )
        ],
    )
