"""Deterministic resolution of cancer names to curated TCGA projects."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oncolncai.schemas import Provenance, ResultStatus, ToolResult


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class MatchMethod(StrEnum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"
    NONE = "none"


class CancerValidationStatus(StrEnum):
    LIVE_VALIDATED = "LIVE_VALIDATED"
    AVAILABLE = "AVAILABLE"
    EXPERIMENTAL = "EXPERIMENTAL"
    UNAVAILABLE = "UNAVAILABLE"


class CancerRegistryEntry(BaseModel):
    """Public read-only view of one canonical cancer/project mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    canonical_name: str
    abbreviation: str
    tcga_project_id: str
    primary_site: str
    tcga_disease_name: str
    synonyms: tuple[str, ...]
    validation_status: CancerValidationStatus
    execution_status: CancerValidationStatus
    expression_source_template: str = "https://gdc.xenahubs.net/download/{project_id}.star_counts.tsv.gz"
    raw_count_source: str = "NCI GDC API: STAR - Counts"
    clinical_source: str = "TCGA PanCanAtlas survival supplement"
    annotation_source: str = "GENCODE v36"
    tumor_sample_type: str = "Primary Tumor (01)"
    normal_sample_type: str = "Solid Tissue Normal (11)"
    unavailable_reason: str | None = None

    @property
    def display_name(self) -> str:
        return f"{self.canonical_name} ({self.tcga_project_id})"


class CancerResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)


class CancerCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_name: str
    abbreviation: str
    tcga_project_id: str
    primary_site: str
    confidence: float = Field(ge=0.0, le=1.0)


class CancerResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    status: ResolutionStatus
    match_method: MatchMethod
    canonical_name: str | None = None
    abbreviation: str | None = None
    tcga_project_id: str | None = None
    synonyms: tuple[str, ...] = ()
    primary_site: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguous: bool
    alternatives: tuple[CancerCandidate, ...] = ()

    @model_validator(mode="after")
    def validate_outcome(self) -> CancerResolution:
        resolved_fields = (
            self.canonical_name,
            self.abbreviation,
            self.tcga_project_id,
            self.primary_site,
        )
        if self.status == ResolutionStatus.RESOLVED:
            if self.ambiguous or any(value is None for value in resolved_fields):
                raise ValueError("resolved result requires identifiers and cannot be ambiguous")
        elif any(value is not None for value in resolved_fields):
            raise ValueError("unresolved result cannot assign cancer identifiers")
        if self.status == ResolutionStatus.AMBIGUOUS:
            if not self.ambiguous or len(self.alternatives) < 2:
                raise ValueError("ambiguous result requires at least two alternatives")
        elif self.ambiguous:
            raise ValueError("only ambiguous results may set ambiguous=true")
        return self


class _CancerRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    canonical_name: str
    abbreviation: str
    tcga_project_id: str
    synonyms: tuple[str, ...]
    primary_site: str
    validation_status: CancerValidationStatus = CancerValidationStatus.EXPERIMENTAL
    execution_status: CancerValidationStatus = CancerValidationStatus.AVAILABLE
    unavailable_reason: str | None = None

    @property
    def aliases(self) -> tuple[str, ...]:
        return (
            self.canonical_name,
            self.abbreviation,
            self.tcga_project_id,
            *self.synonyms,
        )


_CANCERS = (
    _CancerRecord(
        canonical_name="Lung Adenocarcinoma",
        abbreviation="LUAD",
        tcga_project_id="TCGA-LUAD",
        synonyms=(
            "lung adenocarcinoma",
            "adenocarcinoma of the lung",
            "lung adeno carcinoma",
            "lung cancer",
        ),
        primary_site="Lung",
        validation_status=CancerValidationStatus.LIVE_VALIDATED,
    ),
    _CancerRecord(
        canonical_name="Breast Invasive Carcinoma",
        abbreviation="BRCA",
        tcga_project_id="TCGA-BRCA",
        synonyms=("breast cancer", "breast carcinoma", "invasive breast carcinoma"),
        primary_site="Breast",
        validation_status=CancerValidationStatus.AVAILABLE,
    ),
    _CancerRecord(
        canonical_name="Kidney Renal Clear Cell Carcinoma",
        abbreviation="KIRC",
        tcga_project_id="TCGA-KIRC",
        synonyms=(
            "clear cell renal cell carcinoma",
            "renal clear cell carcinoma",
            "ccRCC",
            "kidney cancer",
        ),
        primary_site="Kidney",
        validation_status=CancerValidationStatus.AVAILABLE,
    ),
    _CancerRecord(
        canonical_name="Lung Squamous Cell Carcinoma",
        abbreviation="LUSC",
        tcga_project_id="TCGA-LUSC",
        synonyms=("lung squamous carcinoma", "squamous cell lung cancer", "lung cancer"),
        primary_site="Lung",
    ),
    _CancerRecord(
        canonical_name="Kidney Renal Papillary Cell Carcinoma",
        abbreviation="KIRP",
        tcga_project_id="TCGA-KIRP",
        synonyms=("papillary renal cell carcinoma", "kidney cancer"),
        primary_site="Kidney",
    ),
    _CancerRecord(
        canonical_name="Kidney Chromophobe",
        abbreviation="KICH",
        tcga_project_id="TCGA-KICH",
        synonyms=("chromophobe renal cell carcinoma", "kidney cancer"),
        primary_site="Kidney",
    ),
)

# The public-data clients resolve these projects from the project ID rather than
# from cancer-specific file paths.  LUAD/BRCA/KIRC retain their richer aliases
# above; the remaining canonical TCGA projects use the same generic backend.
_ADDITIONAL_CANCERS = (
    ("Adrenocortical Carcinoma", "ACC", "Adrenal Gland"),
    ("Bladder Urothelial Carcinoma", "BLCA", "Bladder"),
    ("Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma", "CESC", "Cervix"),
    ("Cholangiocarcinoma", "CHOL", "Bile Duct"),
    ("Colon Adenocarcinoma", "COAD", "Colon"),
    ("Lymphoid Neoplasm Diffuse Large B-cell Lymphoma", "DLBC", "Lymph Nodes"),
    ("Esophageal Carcinoma", "ESCA", "Esophagus"),
    ("Glioblastoma Multiforme", "GBM", "Brain"),
    ("Head and Neck Squamous Cell Carcinoma", "HNSC", "Head and Neck"),
    ("Kidney Renal Papillary Cell Carcinoma", "KIRP", "Kidney"),
    ("Acute Myeloid Leukemia", "LAML", "Blood"),
    ("Brain Lower Grade Glioma", "LGG", "Brain"),
    ("Liver Hepatocellular Carcinoma", "LIHC", "Liver"),
    ("Mesothelioma", "MESO", "Pleura"),
    ("Ovarian Serous Cystadenocarcinoma", "OV", "Ovary"),
    ("Pancreatic Adenocarcinoma", "PAAD", "Pancreas"),
    ("Pheochromocytoma and Paraganglioma", "PCPG", "Adrenal Gland"),
    ("Prostate Adenocarcinoma", "PRAD", "Prostate"),
    ("Rectum Adenocarcinoma", "READ", "Rectum"),
    ("Sarcoma", "SARC", "Connective Tissue"),
    ("Skin Cutaneous Melanoma", "SKCM", "Skin"),
    ("Stomach Adenocarcinoma", "STAD", "Stomach"),
    ("Testicular Germ Cell Tumors", "TGCT", "Testis"),
    ("Thyroid Carcinoma", "THCA", "Thyroid"),
    ("Thymoma", "THYM", "Thymus"),
    ("Uterine Corpus Endometrial Carcinoma", "UCEC", "Uterus"),
    ("Uterine Carcinosarcoma", "UCS", "Uterus"),
    ("Uveal Melanoma", "UVM", "Eye"),
)

_EXISTING_IDS = {item.tcga_project_id for item in _CANCERS}
_CANCERS = _CANCERS + tuple(
    _CancerRecord(
        canonical_name=name, abbreviation=abbr, tcga_project_id=f"TCGA-{abbr}",
        synonyms=(name.casefold(),), primary_site=site,
    )
    for name, abbr, site in _ADDITIONAL_CANCERS
    if f"TCGA-{abbr}" not in _EXISTING_IDS
)


def list_supported_cancers() -> tuple[CancerRegistryEntry, ...]:
    """Return the single canonical registry used by resolution and the UI."""

    return tuple(CancerRegistryEntry(
        canonical_name=item.canonical_name, abbreviation=item.abbreviation,
        tcga_project_id=item.tcga_project_id, primary_site=item.primary_site,
        tcga_disease_name=item.canonical_name, synonyms=item.synonyms,
        validation_status=item.validation_status, execution_status=item.execution_status,
        unavailable_reason=item.unavailable_reason,
    ) for item in _CANCERS)


def cancer_from_display_name(value: str) -> CancerRegistryEntry:
    """Resolve a dropdown value without maintaining a second UI mapping."""

    match = next((item for item in list_supported_cancers() if value in (item.display_name, item.tcga_project_id, item.canonical_name)), None)
    if match is None:
        raise ValueError(f"Unsupported TCGA cancer selection: {value}")
    return match


def cancers_mentioned_in_text(text: str) -> tuple[CancerRegistryEntry, ...]:
    """Detect explicit registry aliases embedded in a research question."""

    normalized = f" {_normalize(text)} "
    matches = []
    for item in list_supported_cancers():
        aliases = (item.canonical_name, item.abbreviation, item.tcga_project_id, *item.synonyms)
        if any(f" {_normalize(alias)} " in normalized for alias in aliases if len(_normalize(alias)) >= 4):
            matches.append(item)
    return tuple(matches)


def _normalize(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_value.casefold()).split())


def _candidate(record: _CancerRecord, confidence: float) -> CancerCandidate:
    return CancerCandidate(
        canonical_name=record.canonical_name,
        abbreviation=record.abbreviation,
        tcga_project_id=record.tcga_project_id,
        primary_site=record.primary_site,
        confidence=round(confidence, 3),
    )


def _resolved(
    query: str,
    record: _CancerRecord,
    method: MatchMethod,
    confidence: float,
) -> CancerResolution:
    return CancerResolution(
        query=query,
        status=ResolutionStatus.RESOLVED,
        match_method=method,
        canonical_name=record.canonical_name,
        abbreviation=record.abbreviation,
        tcga_project_id=record.tcga_project_id,
        synonyms=record.synonyms,
        primary_site=record.primary_site,
        confidence=round(confidence, 3),
        ambiguous=False,
    )


def _unresolved(
    query: str,
    status: ResolutionStatus,
    method: MatchMethod,
    candidates: tuple[CancerCandidate, ...],
) -> CancerResolution:
    return CancerResolution(
        query=query,
        status=status,
        match_method=method,
        confidence=candidates[0].confidence if candidates else 0.0,
        ambiguous=status == ResolutionStatus.AMBIGUOUS,
        alternatives=candidates,
    )


def resolve_cancer(
    request: CancerResolveRequest,
) -> ToolResult[CancerResolution]:
    """Resolve a user cancer name without guessing when evidence is ambiguous."""

    query = request.name.strip()
    if not query:
        raise ValueError("cancer name must contain non-whitespace characters")

    exact_matches = [
        record
        for record in _CANCERS
        if any(query == alias for alias in record.aliases)
    ]
    if len(exact_matches) == 1:
        resolution = _resolved(query, exact_matches[0], MatchMethod.EXACT, 1.0)
    elif len(exact_matches) > 1:
        resolution = _unresolved(
            query,
            ResolutionStatus.AMBIGUOUS,
            MatchMethod.EXACT,
            tuple(_candidate(record, 1.0) for record in exact_matches),
        )
    else:
        normalized_query = _normalize(query)
        normalized_matches = [
            record
            for record in _CANCERS
            if any(normalized_query == _normalize(alias) for alias in record.aliases)
        ]
        if len(normalized_matches) == 1:
            resolution = _resolved(query, normalized_matches[0], MatchMethod.NORMALIZED, 0.99)
        elif len(normalized_matches) > 1:
            resolution = _unresolved(
                query,
                ResolutionStatus.AMBIGUOUS,
                MatchMethod.NORMALIZED,
                tuple(_candidate(record, 0.99) for record in normalized_matches),
            )
        else:
            scores: dict[str, float] = defaultdict(float)
            records_by_id = {record.tcga_project_id: record for record in _CANCERS}
            for record in _CANCERS:
                scores[record.tcga_project_id] = max(
                    SequenceMatcher(None, normalized_query, _normalize(alias)).ratio()
                    for alias in record.aliases
                )
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            top_id, top_score = ranked[0]
            second_score = ranked[1][1]
            alternatives = tuple(
                _candidate(records_by_id[project_id], score)
                for project_id, score in ranked[:3]
                if score >= 0.45
            )
            if top_score >= 0.84 and top_score - second_score >= 0.08:
                resolution = _resolved(
                    query, records_by_id[top_id], MatchMethod.FUZZY, top_score
                )
            elif top_score >= 0.72 and len(alternatives) >= 2:
                resolution = _unresolved(
                    query,
                    ResolutionStatus.AMBIGUOUS,
                    MatchMethod.FUZZY,
                    alternatives,
                )
            else:
                resolution = _unresolved(
                    query,
                    ResolutionStatus.UNKNOWN,
                    MatchMethod.NONE,
                    alternatives,
                )

    warning = {
        ResolutionStatus.AMBIGUOUS: "Cancer name is ambiguous; select an alternative.",
        ResolutionStatus.UNKNOWN: "Cancer name could not be resolved confidently.",
    }.get(resolution.status)
    return ToolResult[CancerResolution](
        status=ResultStatus.SUCCESS,
        data=resolution,
        warnings=[] if warning is None else [warning],
        provenance=[
            Provenance(
                source="oncolncai_curated_tcga_mapping",
                dataset_version="1.0",
                query=query,
            )
        ],
    )
