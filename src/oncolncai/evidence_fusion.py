"""Deterministic evidence prioritization across implemented evidence sources."""

from __future__ import annotations

import json
import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oncolncai.providers import LLMProvider
from oncolncai.schemas import Provenance, ResultStatus, ToolResult


class EvidenceDirection(StrEnum):
    UP = "up"
    DOWN = "down"
    ADVERSE = "adverse"
    FAVORABLE = "favorable"
    MIXED = "mixed"
    NONE = "none"


class ConsistencyLabel(StrEnum):
    CONSISTENT = "consistent"
    MOSTLY_CONSISTENT = "mostly_consistent"
    MIXED = "mixed"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"


class LiteratureEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pmid: str = Field(pattern=r"^\d+$")
    provenance: tuple[Provenance, ...] = Field(min_length=1)
    expression_direction: EvidenceDirection | None = None
    prognostic_direction: EvidenceDirection | None = None

    @model_validator(mode="after")
    def validate_directions(self) -> LiteratureEvidenceInput:
        if self.expression_direction not in {None, EvidenceDirection.UP, EvidenceDirection.DOWN, EvidenceDirection.MIXED, EvidenceDirection.NONE}:
            raise ValueError("literature expression direction must be up, down, mixed, none, or missing")
        if self.prognostic_direction not in {None, EvidenceDirection.ADVERSE, EvidenceDirection.FAVORABLE, EvidenceDirection.MIXED, EvidenceDirection.NONE}:
            raise ValueError("literature prognostic direction must be adverse, favorable, mixed, none, or missing")
        return self


class DifferentialExpressionEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    project_id: str = Field(pattern=r"^TCGA-[A-Z0-9]+$")
    log2_fold_change: float = Field(allow_inf_nan=False)
    adjusted_p_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    provenance: tuple[Provenance, ...] = Field(min_length=1)


class SurvivalEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    project_id: str = Field(pattern=r"^TCGA-[A-Z0-9]+$")
    hazard_ratio: float = Field(gt=0, allow_inf_nan=False)
    cox_p_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    log_rank_p_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    provenance: tuple[Provenance, ...] = Field(min_length=1)


class GEOValidationEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    accession: str = Field(pattern=r"^GSE\d+$")
    log2_fold_change: float = Field(allow_inf_nan=False)
    adjusted_p_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    provenance: tuple[Provenance, ...] = Field(min_length=1)


class EvidenceFusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    literature_weight: float = Field(default=0.25, ge=0, allow_inf_nan=False)
    tcga_weight: float = Field(default=0.25, ge=0, allow_inf_nan=False)
    survival_weight: float = Field(default=0.20, ge=0, allow_inf_nan=False)
    geo_weight: float = Field(default=0.20, ge=0, allow_inf_nan=False)
    consistency_weight: float = Field(default=0.10, ge=0, allow_inf_nan=False)
    significance_alpha: float = Field(default=0.05, gt=0, lt=1, allow_inf_nan=False)
    full_expression_effect: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    full_hazard_ratio: float = Field(default=2.0, gt=1, allow_inf_nan=False)
    full_literature_sources: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def require_positive_total_weight(self) -> EvidenceFusionConfig:
        if self.total_weight <= 0:
            raise ValueError("at least one evidence weight must be positive")
        return self

    @property
    def total_weight(self) -> float:
        return self.literature_weight + self.tcga_weight + self.survival_weight + self.geo_weight + self.consistency_weight


class EvidenceFusionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    candidate: str = Field(min_length=1)
    cancer: str = Field(min_length=1)
    literature: tuple[LiteratureEvidenceInput, ...] | None = None
    tcga: DifferentialExpressionEvidenceInput | None = None
    survival: SurvivalEvidenceInput | None = None
    geo: GEOValidationEvidenceInput | None = None
    config: EvidenceFusionConfig = Field(default_factory=EvidenceFusionConfig)

    @model_validator(mode="after")
    def validate_literature_sources(self) -> EvidenceFusionRequest:
        if self.literature is None and self.tcga is None and self.survival is None and self.geo is None:
            raise ValueError("at least one evidence source is required")
        if self.literature is not None:
            pmids = [item.pmid for item in self.literature]
            if not pmids:
                raise ValueError("use null, not an empty collection, for missing literature evidence")
            if len(pmids) != len(set(pmids)):
                raise ValueError("literature PMIDs must be unique")
        return self


class ComponentScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component: str
    available: bool
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    configured_weight: float = Field(ge=0, allow_inf_nan=False)
    weighted_points: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    rationale: str


class ConsistencyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: ConsistencyLabel
    comparable_pairs: int = Field(ge=0)
    agreements: int = Field(ge=0)
    contradictions: int = Field(ge=0)
    details: tuple[str, ...] = ()


class EvidencePrioritizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    cancer: str
    evidence_prioritization_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    available_weight: float = Field(gt=0, allow_inf_nan=False)
    configured_total_weight: float = Field(gt=0, allow_inf_nan=False)
    evidence_coverage: float = Field(gt=0, le=1, allow_inf_nan=False)
    components: tuple[ComponentScore, ...]
    missing_evidence: tuple[str, ...]
    consistency: ConsistencyAssessment
    configuration: EvidenceFusionConfig


class EvidenceSynthesis(BaseModel):
    """Narrative-only model output; numerical fields are intentionally impossible."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    explanation: str = Field(min_length=1, max_length=3000)
    limitations: tuple[str, ...] = ()


def _direction(value: float, positive: EvidenceDirection, negative: EvidenceDirection) -> EvidenceDirection:
    if value > 0:
        return positive
    if value < 0:
        return negative
    return EvidenceDirection.NONE


def _literature_score(items: tuple[LiteratureEvidenceInput, ...], config: EvidenceFusionConfig) -> float:
    return min(len(items) / config.full_literature_sources, 1.0)


def _expression_score(effect: float, p_value: float, config: EvidenceFusionConfig) -> float:
    if p_value > config.significance_alpha:
        return 0.0
    return min(abs(effect) / config.full_expression_effect, 1.0)


def _survival_score(value: SurvivalEvidenceInput, config: EvidenceFusionConfig) -> float:
    if max(value.cox_p_value, value.log_rank_p_value) > config.significance_alpha:
        return 0.0
    return min(abs(math.log(value.hazard_ratio)) / math.log(config.full_hazard_ratio), 1.0)


def _consistency(request: EvidenceFusionRequest) -> tuple[ConsistencyAssessment, float | None]:
    expression: list[tuple[str, EvidenceDirection]] = []
    prognosis: list[tuple[str, EvidenceDirection]] = []
    for item in request.literature or ():
        if item.expression_direction in {EvidenceDirection.UP, EvidenceDirection.DOWN}:
            expression.append((f"PMID:{item.pmid}", item.expression_direction))
        if item.prognostic_direction in {EvidenceDirection.ADVERSE, EvidenceDirection.FAVORABLE}:
            prognosis.append((f"PMID:{item.pmid}", item.prognostic_direction))
    if request.tcga is not None and request.tcga.adjusted_p_value <= request.config.significance_alpha:
        expression.append((request.tcga.project_id, _direction(request.tcga.log2_fold_change, EvidenceDirection.UP, EvidenceDirection.DOWN)))
    if request.geo is not None and request.geo.adjusted_p_value <= request.config.significance_alpha:
        expression.append((request.geo.accession, _direction(request.geo.log2_fold_change, EvidenceDirection.UP, EvidenceDirection.DOWN)))
    if request.survival is not None and max(request.survival.cox_p_value, request.survival.log_rank_p_value) <= request.config.significance_alpha:
        prognosis.append((request.survival.project_id, EvidenceDirection.ADVERSE if request.survival.hazard_ratio > 1 else EvidenceDirection.FAVORABLE))

    details: list[str] = []
    agreements = contradictions = 0
    for dimension, observations in (("expression", expression), ("prognosis", prognosis)):
        for index, left in enumerate(observations):
            for right in observations[index + 1:]:
                agrees = left[1] == right[1]
                agreements += int(agrees)
                contradictions += int(not agrees)
                details.append(f"{dimension}: {left[0]} {left[1].value} vs {right[0]} {right[1].value} ({'agreement' if agrees else 'contradiction'})")
    pairs = agreements + contradictions
    if pairs == 0:
        return ConsistencyAssessment(label=ConsistencyLabel.INSUFFICIENT, comparable_pairs=0, agreements=0, contradictions=0), None
    score = agreements / pairs
    if score == 1:
        label = ConsistencyLabel.CONSISTENT
    elif score >= 0.75:
        label = ConsistencyLabel.MOSTLY_CONSISTENT
    elif score >= 0.5:
        label = ConsistencyLabel.MIXED
    else:
        label = ConsistencyLabel.CONFLICTING
    return ConsistencyAssessment(label=label, comparable_pairs=pairs, agreements=agreements, contradictions=contradictions, details=tuple(details)), score


def prioritize_evidence(request: EvidenceFusionRequest) -> ToolResult[EvidencePrioritizationResult]:
    consistency, consistency_score = _consistency(request)
    raw = {
        "literature": (_literature_score(request.literature, request.config), request.config.literature_weight, f"{len(request.literature)} unique PMID(s); full support at {request.config.full_literature_sources}") if request.literature is not None else (None, request.config.literature_weight, "missing"),
        "tcga_differential_expression": (_expression_score(request.tcga.log2_fold_change, request.tcga.adjusted_p_value, request.config), request.config.tcga_weight, "adjusted p-value and absolute log2 fold-change") if request.tcga is not None else (None, request.config.tcga_weight, "missing"),
        "survival": (_survival_score(request.survival, request.config), request.config.survival_weight, "Cox and log-rank significance with hazard-ratio magnitude") if request.survival is not None else (None, request.config.survival_weight, "missing"),
        "geo_replication": (_expression_score(request.geo.log2_fold_change, request.geo.adjusted_p_value, request.config), request.config.geo_weight, "adjusted p-value and absolute log2 fold-change") if request.geo is not None else (None, request.config.geo_weight, "missing"),
        "cross_source_consistency": (consistency_score, request.config.consistency_weight, f"{consistency.agreements} agreement(s), {consistency.contradictions} contradiction(s)"),
    }
    components = tuple(ComponentScore(component=name, available=score is not None, score=score, configured_weight=weight, weighted_points=None if score is None else score * weight, rationale=rationale) for name, (score, weight, rationale) in raw.items())
    available_weight = sum(item.configured_weight for item in components if item.available)
    if available_weight <= 0:
        raise ValueError("no weighted evidence component is available")
    numerator = sum(item.weighted_points or 0 for item in components)
    score = round(100 * numerator / available_weight, 6)
    missing = tuple(item.component for item in components if not item.available)
    provenance = [source for item in request.literature or () for source in item.provenance]
    for value in (request.tcga, request.survival, request.geo):
        if value is not None:
            provenance.extend(value.provenance)
    unique = list({item.model_dump_json(): item for item in provenance}.values())
    result = EvidencePrioritizationResult(
        candidate=request.candidate,
        cancer=request.cancer,
        evidence_prioritization_score=score,
        available_weight=available_weight,
        configured_total_weight=request.config.total_weight,
        evidence_coverage=min(available_weight / request.config.total_weight, 1.0),
        components=components,
        missing_evidence=missing,
        consistency=consistency,
        configuration=request.config,
    )
    warnings = [f"Missing evidence preserved: {', '.join(missing)}"] if missing else []
    return ToolResult(status=ResultStatus.SUCCESS, data=result, provenance=unique, warnings=warnings)


def synthesize_evidence_prioritization(result: EvidencePrioritizationResult, *, provider: LLMProvider) -> EvidenceSynthesis:
    """Explain an immutable deterministic result without exposing score fields for mutation."""
    prompt = "Explain this deterministic evidence prioritization result without changing or reframing its numbers as probability: " + json.dumps(result.model_dump(mode="json"), sort_keys=True)
    return EvidenceSynthesis.model_validate(provider.generate_structured(prompt=prompt, output_schema=EvidenceSynthesis))
