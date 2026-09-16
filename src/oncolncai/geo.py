"""Deterministic-first GEO discovery and validation-dataset suitability."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator

from oncolncai.providers import LLMProvider
from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult


EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class GEOAssay(StrEnum):
    RNA_SEQ = "rna_seq"
    MICROARRAY = "microarray"
    OTHER = "other"
    UNKNOWN = "unknown"


class GEOAnnotation(StrEnum):
    ENSEMBL_GENE = "ensembl_gene"
    GENE_SYMBOL = "gene_symbol"
    PROBE_TO_GENE_MAPPING = "probe_to_gene_mapping"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class CriterionStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    AMBIGUOUS = "ambiguous"


class GEOSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    cancer: str = Field(min_length=1, max_length=200)
    tissue: str = Field(min_length=1, max_length=200)
    disease_aliases: tuple[str, ...] = ()
    tissue_aliases: tuple[str, ...] = ()
    allowed_assays: tuple[GEOAssay, ...] = (GEOAssay.RNA_SEQ, GEOAssay.MICROARRAY)
    min_cases: int = Field(default=10, ge=1)
    min_controls: int = Field(default=10, ge=1)
    require_lncrna_annotation: bool = True
    max_results: int = Field(default=20, ge=1, le=100)

    @field_validator("disease_aliases", "tissue_aliases")
    @classmethod
    def no_blank_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("aliases cannot contain blank values")
        return values


class GEOSeriesMetadata(BaseModel):
    """Normalized metadata; fields that GEO cannot establish remain explicit unknowns."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    accession: str = Field(pattern=r"^GSE\d+$")
    title: str = Field(min_length=1)
    summary: str = ""
    organism: str | None = None
    disease_terms: tuple[str, ...] = ()
    tissue_terms: tuple[str, ...] = ()
    assay: GEOAssay = GEOAssay.UNKNOWN
    platform_accessions: tuple[str, ...] = ()
    case_sample_count: int | None = Field(default=None, ge=0)
    control_sample_count: int | None = Field(default=None, ge=0)
    annotation: GEOAnnotation = GEOAnnotation.UNKNOWN
    metadata_url: str = Field(min_length=1)
    retrieved_at: datetime


class GEOBackend(Protocol):
    def search(self, query: str, max_results: int, timeout: float) -> tuple[GEOSeriesMetadata, ...]: ...


def build_geo_query(request: GEOSearchRequest) -> str:
    cancer = request.cancer.replace('"', "")
    tissue = request.tissue.replace('"', "")
    return f'("{cancer}"[All Fields]) AND ("{tissue}"[All Fields]) AND gse[Entry Type]'


class NCBIGEOBackend:
    """Bounded NCBI GEO DataSets ESearch/ESummary backend.

    ESummary does not reliably expose experimental group counts or lncRNA
    annotation compatibility. Those values deliberately remain unknown so the
    suitability policy fails closed instead of inferring them from a title.
    """

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key.strip() if api_key else None

    def _get(self, endpoint: str, params: dict[str, str], timeout: float) -> dict:
        values = dict(params)
        if self._api_key:
            values["api_key"] = self._api_key
        url = f"{EUTILS_BASE_URL}/{endpoint}?{urlencode(values)}"
        request = Request(url, headers={"User-Agent": "OncoLncAI/0.1"})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read())

    def search(self, query: str, max_results: int, timeout: float) -> tuple[GEOSeriesMetadata, ...]:
        found = self._get(
            "esearch.fcgi", {"db": "gds", "term": query, "retmode": "json", "retmax": str(max_results)}, timeout
        )
        ids = tuple(dict.fromkeys(found["esearchresult"].get("idlist", ())))
        if not ids:
            return ()
        summaries = self._get(
            "esummary.fcgi", {"db": "gds", "id": ",".join(ids), "retmode": "json"}, timeout
        )["result"]
        retrieved_at = datetime.now(timezone.utc)
        records: list[GEOSeriesMetadata] = []
        for uid in ids:
            item = summaries.get(str(uid), {})
            accession = str(item.get("accession", ""))
            if not re.fullmatch(r"GSE\d+", accession):
                continue
            gpl = item.get("gpl", ())
            if isinstance(gpl, str):
                gpl = re.findall(r"GPL\d+", gpl)
            records.append(
                GEOSeriesMetadata(
                    accession=accession,
                    title=str(item.get("title") or accession),
                    summary=str(item.get("summary") or ""),
                    organism=str(item.get("taxon") or "") or None,
                    platform_accessions=tuple(str(value) for value in gpl if re.fullmatch(r"GPL\d+", str(value))),
                    metadata_url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
                    retrieved_at=retrieved_at,
                )
            )
        return tuple(records)


class GEOSemanticDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    accession: str = Field(pattern=r"^GSE\d+$")
    disease_match: bool
    tissue_match: bool
    explanation: str = Field(min_length=1, max_length=500)


class GEOSemanticBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    decisions: tuple[GEOSemanticDecision, ...]


class GEOCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    criterion: str
    status: CriterionStatus
    reason: str


class GEODatasetAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset: GEOSeriesMetadata
    criteria: tuple[GEOCriterion, ...]
    suitable: bool
    score: int = Field(ge=0, le=6)
    semantic_review_used: bool = False


class GEOSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str
    assessments: tuple[GEODatasetAssessment, ...]
    suitable_datasets: tuple[GEODatasetAssessment, ...]
    no_suitable_validation_dataset: bool


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _term_status(observed: tuple[str, ...], expected: tuple[str, ...]) -> CriterionStatus:
    normalized_observed = {_normalize(value) for value in observed if _normalize(value)}
    normalized_expected = {_normalize(value) for value in expected if _normalize(value)}
    if not normalized_observed:
        return CriterionStatus.AMBIGUOUS
    if any(a in b or b in a for a in normalized_observed for b in normalized_expected):
        return CriterionStatus.PASS
    return CriterionStatus.FAIL


def _criterion(name: str, status: CriterionStatus, reason: str) -> GEOCriterion:
    return GEOCriterion(criterion=name, status=status, reason=reason)


class GEOClient:
    def __init__(
        self,
        *,
        backend: GEOBackend | None = None,
        semantic_provider: LLMProvider | None = None,
        timeout: float = 20.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._backend = backend or NCBIGEOBackend()
        self._semantic_provider = semantic_provider
        self._timeout = timeout

    def search(self, request: GEOSearchRequest) -> ToolResult[GEOSearchResult]:
        query = build_geo_query(request)
        try:
            records = self._backend.search(query, request.max_results, self._timeout)
            decisions = self._semantic_decisions(request, records)
            assessments = tuple(self._assess(request, record, decisions.get(record.accession)) for record in records)
            ranked = tuple(sorted((item for item in assessments if item.suitable), key=lambda x: (-x.score, -(x.dataset.case_sample_count or 0) - (x.dataset.control_sample_count or 0), x.dataset.accession)))
            data = GEOSearchResult(
                query=query,
                assessments=assessments,
                suitable_datasets=ranked,
                no_suitable_validation_dataset=not ranked,
            )
            provenance = [Provenance(source="NCBI GEO DataSets", source_id=item.accession, query=query) for item in records]
            warnings = ["No suitable GEO validation dataset met every required criterion."] if not ranked else []
            return ToolResult(status=ResultStatus.SUCCESS, data=data, provenance=provenance, warnings=warnings)
        except Exception as exc:
            return ToolResult(
                status=ResultStatus.FAILURE,
                error=ToolError(code="geo_search_failed", message=str(exc) or type(exc).__name__, retryable=True),
                provenance=[Provenance(source="NCBI GEO DataSets", query=query)],
            )

    def _semantic_decisions(self, request: GEOSearchRequest, records: tuple[GEOSeriesMetadata, ...]) -> dict[str, GEOSemanticDecision]:
        ambiguous = [record for record in records if _term_status(record.disease_terms, (request.cancer, *request.disease_aliases)) == CriterionStatus.AMBIGUOUS or _term_status(record.tissue_terms, (request.tissue, *request.tissue_aliases)) == CriterionStatus.AMBIGUOUS]
        if not ambiguous or self._semantic_provider is None:
            return {}
        payload = [{"accession": item.accession, "title": item.title, "summary": item.summary} for item in ambiguous]
        raw = self._semantic_provider.generate_structured(
            prompt=f"Resolve only disease and tissue semantic ambiguity for {request.cancer} / {request.tissue}. Candidates: {json.dumps(payload)}",
            output_schema=GEOSemanticBatch,
        )
        batch = GEOSemanticBatch.model_validate(raw)
        expected = {item.accession for item in ambiguous}
        actual = [item.accession for item in batch.decisions]
        if set(actual) != expected or len(actual) != len(set(actual)):
            raise ValueError("semantic GEO decisions must cover each ambiguous accession exactly once")
        return {item.accession: item for item in batch.decisions}

    def _assess(self, request: GEOSearchRequest, record: GEOSeriesMetadata, semantic: GEOSemanticDecision | None) -> GEODatasetAssessment:
        disease = _term_status(record.disease_terms, (request.cancer, *request.disease_aliases))
        tissue = _term_status(record.tissue_terms, (request.tissue, *request.tissue_aliases))
        if semantic is not None:
            if disease == CriterionStatus.AMBIGUOUS:
                disease = CriterionStatus.PASS if semantic.disease_match else CriterionStatus.FAIL
            if tissue == CriterionStatus.AMBIGUOUS:
                tissue = CriterionStatus.PASS if semantic.tissue_match else CriterionStatus.FAIL
        assay = CriterionStatus.PASS if record.assay in request.allowed_assays and record.platform_accessions else CriterionStatus.FAIL
        groups = CriterionStatus.PASS if (record.case_sample_count or 0) > 0 and (record.control_sample_count or 0) > 0 else CriterionStatus.FAIL
        sample_size = CriterionStatus.PASS if (record.case_sample_count or 0) >= request.min_cases and (record.control_sample_count or 0) >= request.min_controls else CriterionStatus.FAIL
        compatible = record.annotation in {GEOAnnotation.ENSEMBL_GENE, GEOAnnotation.GENE_SYMBOL, GEOAnnotation.PROBE_TO_GENE_MAPPING}
        annotation = CriterionStatus.PASS if (compatible or not request.require_lncrna_annotation) else CriterionStatus.FAIL
        values = (
            _criterion("cancer_disease_match", disease, "structured disease metadata matches" if disease == CriterionStatus.PASS else "disease match is absent or unresolved"),
            _criterion("tissue_match", tissue, "structured tissue metadata matches" if tissue == CriterionStatus.PASS else "tissue match is absent or unresolved"),
            _criterion("assay_platform", assay, f"assay={record.assay.value}; platforms={','.join(record.platform_accessions) or 'unknown'}"),
            _criterion("case_control_availability", groups, f"cases={record.case_sample_count}; controls={record.control_sample_count}"),
            _criterion("sample_size", sample_size, f"requires cases>={request.min_cases}, controls>={request.min_controls}"),
            _criterion("annotation_compatibility", annotation, f"annotation={record.annotation.value}"),
        )
        score = sum(value.status == CriterionStatus.PASS for value in values)
        return GEODatasetAssessment(dataset=record, criteria=values, suitable=score == len(values), score=score, semantic_review_used=semantic is not None)


def search_geo(request: GEOSearchRequest, *, client: GEOClient | None = None) -> ToolResult[GEOSearchResult]:
    return (client or GEOClient()).search(request)
