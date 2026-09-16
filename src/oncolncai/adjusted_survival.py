"""Deterministic multivariable Cox PH models and candidate screening."""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import norm, spearmanr

from oncolncai.differential_expression import _benjamini_hochberg


class ConvergenceStatus(StrEnum):
    CONVERGED = "CONVERGED"
    FAILED = "FAILED"


class AdjustedSurvivalSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    patient_id: str
    time: float = Field(gt=0, allow_inf_nan=False)
    event: int = Field(ge=0, le=1)
    expression: float = Field(allow_inf_nan=False)
    age: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    sex: str | None = None
    pathological_stage: str | None = None


class AdjustedCoxRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project_id: str
    gene: str
    samples: tuple[AdjustedSurvivalSample, ...] = Field(min_length=10)
    include_age: bool = True
    include_sex: bool = True
    include_stage: bool = True
    minimum_events: int = Field(default=10, ge=3)
    minimum_events_per_parameter: float = Field(default=5.0, ge=1)

    @model_validator(mode="after")
    def validate_samples(self) -> "AdjustedCoxRequest":
        ids = [item.patient_id for item in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("patient IDs must be unique")
        if sum(item.event for item in self.samples) < self.minimum_events:
            raise ValueError(f"adjusted Cox requires at least {self.minimum_events} events")
        return self


class PHDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    term: str
    statistic: float | None = None
    p_value: float | None = Field(default=None, ge=0, le=1)
    violation: bool | None = None


class AdjustedCoxResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project_id: str
    gene: str
    sample_count: int
    initial_survival_eligible_count: int
    complete_case_count: int
    event_count: int
    adjusted_hazard_ratio: float = Field(gt=0)
    confidence_interval_lower: float = Field(gt=0)
    confidence_interval_upper: float = Field(gt=0)
    p_value: float = Field(ge=0, le=1)
    fdr: float | None = Field(default=None, ge=0, le=1)
    covariates_used: tuple[str, ...]
    excluded_missing_covariates: int
    missingness_by_covariate: dict[str, int]
    model_parameter_count: int
    events_per_parameter: float
    convergence_status: ConvergenceStatus
    diagnostics: tuple[PHDiagnostic, ...]
    overall_ph_violation: bool | None


def normalize_stage(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().upper().replace("STAGE", "").strip()
    text = text.replace(" ", "")
    match = next((stage for stage in ("IV", "III", "II", "I") if text.startswith(stage)), None)
    return match


def _design(request: AdjustedCoxRequest) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], int, dict[str, int]]:
    usable: list[AdjustedSurvivalSample] = []
    stages: set[str] = set()
    sexes: set[str] = set()
    missingness = {"age": 0, "sex": 0, "pathological_stage": 0}
    for item in request.samples:
        stage = normalize_stage(item.pathological_stage)
        missingness["age"] += int(request.include_age and item.age is None)
        missingness["sex"] += int(request.include_sex and not item.sex)
        missingness["pathological_stage"] += int(request.include_stage and stage is None)
        missing = ((request.include_age and item.age is None) or
                   (request.include_sex and not item.sex) or
                   (request.include_stage and stage is None))
        if missing:
            continue
        usable.append(item)
        if item.sex:
            sexes.add(item.sex.strip().upper())
        if stage:
            stages.add(stage)
    if len(usable) < 10 or sum(item.event for item in usable) < request.minimum_events:
        raise ValueError("insufficient complete cases/events for adjusted Cox")
    columns = ["expression_z"]
    sex_levels = sorted(sexes)[1:] if request.include_sex else []
    stage_levels = sorted(stages, key=("I", "II", "III", "IV").index)[1:] if request.include_stage else []
    if request.include_age:
        columns.append("age_z")
    columns.extend(f"sex[{level}]" for level in sex_levels)
    columns.extend(f"stage[{level}]" for level in stage_levels)
    expression = np.asarray([item.expression for item in usable], dtype=float)
    if float(np.std(expression)) <= 1e-12:
        raise ValueError("candidate expression has no variance")
    age_values = np.asarray([item.age for item in usable], dtype=float) if request.include_age else None
    vectors: list[list[float]] = []
    for index, item in enumerate(usable):
        row = [(item.expression - float(np.mean(expression))) / float(np.std(expression))]
        if age_values is not None:
            std = float(np.std(age_values))
            if std <= 1e-12:
                raise ValueError("age covariate has no variance")
            row.append((float(item.age) - float(np.mean(age_values))) / std)
        sex = item.sex.strip().upper() if item.sex else ""
        stage = normalize_stage(item.pathological_stage)
        row.extend(float(sex == level) for level in sex_levels)
        row.extend(float(stage == level) for level in stage_levels)
        vectors.append(row)
    matrix = np.asarray(vectors, dtype=float)
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError("adjusted Cox design matrix is singular")
    events = np.asarray([item.event for item in usable])
    events_per_parameter = float(np.sum(events)) / matrix.shape[1]
    if events_per_parameter < request.minimum_events_per_parameter:
        raise ValueError(
            f"insufficient events/model complexity: {int(np.sum(events))} events for "
            f"{matrix.shape[1]} parameters ({events_per_parameter:.2f} per parameter; "
            f"requires {request.minimum_events_per_parameter:.2f})"
        )
    return (np.asarray([item.time for item in usable]), events, matrix, tuple(columns),
            len(request.samples) - len(usable), missingness)


def _fit_cox(times: np.ndarray, events: np.ndarray, matrix: np.ndarray,
             max_iterations: int = 100, tolerance: float = 1e-8) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float]]:
    beta = np.zeros(matrix.shape[1])
    for _ in range(max_iterations):
        eta = np.clip(matrix @ beta, -40, 40)
        weights = np.exp(eta)
        score = np.zeros_like(beta)
        information = np.zeros((len(beta), len(beta)))
        for index in np.where(events == 1)[0]:
            risk = times >= times[index]
            total = float(np.sum(weights[risk]))
            mean = np.sum(matrix[risk] * weights[risk, None], axis=0) / total
            second = (matrix[risk].T * weights[risk]) @ matrix[risk] / total
            score += matrix[index] - mean
            information += second - np.outer(mean, mean)
        try:
            delta = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("adjusted Cox information matrix is singular") from exc
        beta += delta
        if float(np.max(np.abs(delta))) < tolerance:
            break
    else:
        raise ValueError("adjusted Cox model did not converge")
    covariance = np.linalg.inv(information)
    residuals: list[np.ndarray] = []
    event_times: list[float] = []
    weights = np.exp(np.clip(matrix @ beta, -40, 40))
    for index in np.where(events == 1)[0]:
        risk = times >= times[index]
        mean = np.sum(matrix[risk] * weights[risk, None], axis=0) / float(np.sum(weights[risk]))
        residuals.append(matrix[index] - mean)
        event_times.append(float(times[index]))
    return beta, covariance, residuals, event_times


def run_adjusted_cox(request: AdjustedCoxRequest) -> AdjustedCoxResult:
    times, events, matrix, columns, excluded, missingness = _design(request)
    beta, covariance, residuals, event_times = _fit_cox(times, events, matrix)
    standard_error = math.sqrt(float(covariance[0, 0]))
    p_value = float(2 * norm.sf(abs(float(beta[0] / standard_error))))
    diagnostics: list[PHDiagnostic] = []
    residual_matrix = np.asarray(residuals)
    log_times = np.log(np.asarray(event_times))
    for index, name in enumerate(columns):
        if len(log_times) < 3 or float(np.std(residual_matrix[:, index])) <= 1e-12:
            diagnostics.append(PHDiagnostic(term=name))
            continue
        statistic, diagnostic_p = spearmanr(log_times, residual_matrix[:, index])
        diagnostics.append(PHDiagnostic(term=name, statistic=float(statistic), p_value=float(diagnostic_p), violation=bool(diagnostic_p < 0.05)))
    violations = [item.violation for item in diagnostics if item.violation is not None]
    return AdjustedCoxResult(
        project_id=request.project_id, gene=request.gene, sample_count=len(times),
        initial_survival_eligible_count=len(request.samples), complete_case_count=len(times), event_count=int(np.sum(events)),
        adjusted_hazard_ratio=float(math.exp(beta[0])),
        confidence_interval_lower=float(math.exp(beta[0] - 1.96 * standard_error)),
        confidence_interval_upper=float(math.exp(beta[0] + 1.96 * standard_error)), p_value=p_value,
        covariates_used=columns, excluded_missing_covariates=excluded,
        missingness_by_covariate=missingness, model_parameter_count=matrix.shape[1],
        events_per_parameter=float(np.sum(events)) / matrix.shape[1],
        convergence_status=ConvergenceStatus.CONVERGED, diagnostics=tuple(diagnostics),
        overall_ph_violation=any(violations) if violations else None,
    )


def apply_adjusted_cox_fdr(results: tuple[AdjustedCoxResult, ...]) -> tuple[AdjustedCoxResult, ...]:
    adjusted = _benjamini_hochberg([item.p_value for item in results])
    return tuple(item.model_copy(update={"fdr": fdr}) for item, fdr in zip(results, adjusted))
