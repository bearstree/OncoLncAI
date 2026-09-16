"""Deterministic Kaplan-Meier, log-rank, and univariate Cox PH analysis."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import chi2, norm, spearmanr

from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult


class SurvivalEndpoint(StrEnum):
    OS = "OS"
    DSS = "DSS"
    PFI = "PFI"


class SurvivalSample(BaseModel):
    """One aligned expression and survival observation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    sample_id: str = Field(min_length=1)
    time: float = Field(gt=0, allow_inf_nan=False)
    event: int = Field(ge=0, le=1)
    expression: float = Field(allow_inf_nan=False)


class SurvivalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    project_id: str = Field(pattern=r"^TCGA-[A-Z0-9]+$")
    gene: str = Field(min_length=1)
    samples: tuple[SurvivalSample, ...] = Field(min_length=4)
    endpoint: SurvivalEndpoint = SurvivalEndpoint.OS
    grouping_method: str = Field(default="median", pattern=r"^median$")
    alpha: Literal[0.05] = 0.05
    min_samples: int = Field(default=20, ge=4)
    min_events: int = Field(default=5, ge=1)
    cohort_source: str = Field(default="GDC clinical", min_length=1)
    expression_source: str = Field(default="GDC RNA-seq", min_length=1)

    @model_validator(mode="after")
    def validate_analysis_input(self) -> SurvivalRequest:
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("sample IDs must be unique")
        if len(self.samples) < self.min_samples:
            raise ValueError(
                f"survival analysis requires at least {self.min_samples} samples"
            )
        events = sum(sample.event for sample in self.samples)
        if events < self.min_events:
            raise ValueError(
                f"survival analysis requires at least {self.min_events} events"
            )
        cutoff = float(np.median([sample.expression for sample in self.samples]))
        groups = [sample.expression > cutoff for sample in self.samples]
        if all(groups) or not any(groups):
            raise ValueError("median grouping produced an empty expression group")
        if min(sum(groups), len(groups) - sum(groups)) < 2:
            raise ValueError("each expression group requires at least two samples")
        return self


class KaplanMeierPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    time: float = Field(ge=0)
    at_risk: int = Field(ge=0)
    events: int = Field(ge=0)
    censored: int = Field(ge=0)
    survival_probability: float = Field(ge=0, le=1)


class SurvivalGroupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    sample_count: int = Field(ge=1)
    event_count: int = Field(ge=0)
    curve: tuple[KaplanMeierPoint, ...]


class SurvivalAnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: SurvivalEndpoint
    grouping_method: str
    expression_cutoff: float
    high_group_rule: str
    cox_covariate: str
    ties_method: str
    alpha: float
    min_samples: int
    min_events: int


class CoxDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    converged: bool
    iterations: int = Field(ge=0)
    coefficient: float = Field(allow_inf_nan=False)
    standard_error: float = Field(gt=0, allow_inf_nan=False)
    proportional_hazards_test: str = "Spearman correlation of Schoenfeld residuals with log event time"
    proportional_hazards_p_value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    proportional_hazards_assumption_met: bool | None = None


class SurvivalAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    gene: str
    sample_count: int = Field(ge=1)
    event_count: int = Field(ge=1)
    hazard_ratio: float = Field(gt=0, allow_inf_nan=False)
    confidence_interval_lower: float = Field(gt=0, allow_inf_nan=False)
    confidence_interval_upper: float = Field(gt=0, allow_inf_nan=False)
    cox_p_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    log_rank_p_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    groups: tuple[SurvivalGroupResult, SurvivalGroupResult]
    configuration: SurvivalAnalysisConfig
    diagnostics: CoxDiagnostics


def _kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[KaplanMeierPoint, ...]:
    survival = 1.0
    points = [KaplanMeierPoint(time=0.0, at_risk=len(times), events=0, censored=0, survival_probability=1.0)]
    for time in np.unique(times):
        at_risk = int(np.sum(times >= time))
        at_time = times == time
        event_count = int(np.sum(events[at_time]))
        censored = int(np.sum(at_time) - event_count)
        if event_count:
            survival *= 1.0 - event_count / at_risk
        points.append(
            KaplanMeierPoint(
                time=float(time),
                at_risk=at_risk,
                events=event_count,
                censored=censored,
                survival_probability=float(survival),
            )
        )
    return tuple(points)


def _log_rank(times: np.ndarray, events: np.ndarray, high: np.ndarray) -> float:
    observed_minus_expected = 0.0
    variance = 0.0
    for time in np.unique(times[events == 1]):
        risk = times >= time
        at_time = (times == time) & (events == 1)
        n = int(np.sum(risk))
        n_high = int(np.sum(risk & high))
        deaths = int(np.sum(at_time))
        deaths_high = int(np.sum(at_time & high))
        observed_minus_expected += deaths_high - deaths * n_high / n
        if n > 1:
            variance += deaths * n_high * (n - n_high) * (n - deaths) / (n * n * (n - 1))
    if variance <= 0:
        raise ValueError("log-rank variance is zero; groups cannot be compared")
    statistic = observed_minus_expected**2 / variance
    return float(chi2.sf(statistic, df=1))


def _cox_binary(
    times: np.ndarray, events: np.ndarray, high: np.ndarray, alpha: float
) -> tuple[float, float, float, float, float, CoxDiagnostics]:
    x = high.astype(float)
    beta = 0.0
    converged = False
    iterations = 0
    information = 0.0
    for iterations in range(1, 51):
        weights = np.exp(np.clip(beta * x, -50, 50))
        score = 0.0
        information = 0.0
        for time in np.unique(times[events == 1]):
            event_mask = (times == time) & (events == 1)
            risk_mask = times >= time
            deaths = int(np.sum(event_mask))
            risk_weights = weights[risk_mask]
            risk_x = x[risk_mask]
            denominator = float(np.sum(risk_weights))
            mean_x = float(np.sum(risk_weights * risk_x) / denominator)
            score += float(np.sum(x[event_mask])) - deaths * mean_x
            information += deaths * (
                float(np.sum(risk_weights * risk_x * risk_x) / denominator) - mean_x**2
            )
        if information <= 1e-12:
            raise ValueError("Cox information matrix is singular")
        change = score / information
        beta += change
        if abs(change) < 1e-9:
            converged = True
            break
    if not converged or not math.isfinite(beta):
        raise ValueError("Cox model did not converge")
    standard_error = math.sqrt(1.0 / information)
    z_value = beta / standard_error
    p_value = float(2 * norm.sf(abs(z_value)))
    critical = float(norm.ppf(1 - alpha / 2))
    hazard_ratio = math.exp(beta)
    lower = math.exp(beta - critical * standard_error)
    upper = math.exp(beta + critical * standard_error)
    if not all(math.isfinite(value) and value > 0 for value in (hazard_ratio, lower, upper)):
        raise ValueError("Cox model produced non-finite estimates")
    ph_p_value = _proportional_hazards_test(times, events, x, beta)
    return hazard_ratio, lower, upper, p_value, beta, CoxDiagnostics(
        converged=True,
        iterations=iterations,
        coefficient=beta,
        standard_error=standard_error,
        proportional_hazards_p_value=ph_p_value,
        proportional_hazards_assumption_met=(ph_p_value >= alpha) if ph_p_value is not None else None,
    )


def _proportional_hazards_test(
    times: np.ndarray, events: np.ndarray, covariate: np.ndarray, beta: float
) -> float | None:
    """Test time trend in Schoenfeld residuals for the fitted binary Cox term."""

    residuals: list[float] = []
    event_times: list[float] = []
    weights = np.exp(np.clip(beta * covariate, -50, 50))
    for time in np.unique(times[events == 1]):
        risk = times >= time
        expected = float(np.sum(weights[risk] * covariate[risk]) / np.sum(weights[risk]))
        for observed in covariate[(times == time) & (events == 1)]:
            residuals.append(float(observed - expected))
            event_times.append(float(time))
    if len(residuals) < 5 or np.ptp(residuals) <= 1e-12 or np.ptp(event_times) <= 1e-12:
        return None
    result = spearmanr(np.log(event_times), residuals)
    p_value = float(result.pvalue)
    return p_value if math.isfinite(p_value) else None


def run_survival_analysis(
    request: SurvivalRequest,
) -> ToolResult[SurvivalAnalysisResult]:
    """Run median-split KM, log-rank, and Cox PH without model-generated numbers."""

    times = np.asarray([sample.time for sample in request.samples], dtype=float)
    events = np.asarray([sample.event for sample in request.samples], dtype=int)
    expression = np.asarray([sample.expression for sample in request.samples], dtype=float)
    cutoff = float(np.median(expression))
    high = expression > cutoff
    try:
        log_rank_p = _log_rank(times, events, high)
        hazard_ratio, lower, upper, cox_p, _, diagnostics = _cox_binary(
            times, events, high, request.alpha
        )
    except ValueError as exc:
        return ToolResult[SurvivalAnalysisResult](
            status=ResultStatus.FAILURE,
            error=ToolError(code="survival_model_failed", message=str(exc), retryable=False),
            provenance=_provenance(request),
        )

    low_group = SurvivalGroupResult(
        label="low",
        sample_count=int(np.sum(~high)),
        event_count=int(np.sum(events[~high])),
        curve=_kaplan_meier(times[~high], events[~high]),
    )
    high_group = SurvivalGroupResult(
        label="high",
        sample_count=int(np.sum(high)),
        event_count=int(np.sum(events[high])),
        curve=_kaplan_meier(times[high], events[high]),
    )
    result = SurvivalAnalysisResult(
        project_id=request.project_id,
        gene=request.gene,
        sample_count=len(request.samples),
        event_count=int(np.sum(events)),
        hazard_ratio=hazard_ratio,
        confidence_interval_lower=lower,
        confidence_interval_upper=upper,
        cox_p_value=cox_p,
        log_rank_p_value=log_rank_p,
        groups=(low_group, high_group),
        configuration=SurvivalAnalysisConfig(
            endpoint=request.endpoint,
            grouping_method=request.grouping_method,
            expression_cutoff=cutoff,
            high_group_rule="expression > median; ties assigned to low",
            cox_covariate="high-expression indicator (high vs low)",
            ties_method="Breslow",
            alpha=request.alpha,
            min_samples=request.min_samples,
            min_events=request.min_events,
        ),
        diagnostics=diagnostics,
    )
    warnings = []
    if diagnostics.proportional_hazards_assumption_met is False:
        warnings.append(
            "The proportional-hazards diagnostic was significant; interpret the constant Cox HR cautiously."
        )
    if diagnostics.proportional_hazards_assumption_met is None:
        warnings.append("The proportional-hazards diagnostic was not estimable from these events.")
    return ToolResult[SurvivalAnalysisResult](
        status=ResultStatus.SUCCESS,
        data=result,
        warnings=warnings,
        provenance=_provenance(request),
    )


def _provenance(request: SurvivalRequest) -> list[Provenance]:
    return [
        Provenance(source=request.cohort_source, source_id=request.project_id),
        Provenance(source=request.expression_source, source_id=request.gene),
    ]
