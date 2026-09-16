import numpy as np
import pytest

from oncolncai.adjusted_survival import AdjustedCoxRequest, AdjustedSurvivalSample, apply_adjusted_cox_fdr, normalize_stage, run_adjusted_cox


def _samples() -> tuple[AdjustedSurvivalSample, ...]:
    rng = np.random.default_rng(12)
    expression = rng.normal(size=80)
    times = np.maximum(1, 500 - expression * 90 + rng.normal(0, 80, 80))
    events = rng.binomial(1, 0.65, 80)
    return tuple(AdjustedSurvivalSample(
        patient_id=f"P{i}", time=float(times[i]), event=int(events[i]), expression=float(expression[i]),
        age=float(45 + i % 30), sex="MALE" if i % 2 else "FEMALE",
        pathological_stage=f"Stage {('I', 'II', 'III')[i % 3]}",
    ) for i in range(80))


def test_adjusted_cox_reports_covariates_and_ph_diagnostics() -> None:
    result = run_adjusted_cox(AdjustedCoxRequest(project_id="TCGA-LUAD", gene="LNC1", samples=_samples()))
    assert result.sample_count == 80
    assert result.event_count == sum(item.event for item in _samples())
    assert result.convergence_status == "CONVERGED"
    assert result.covariates_used[0] == "expression_z"
    assert {item.term for item in result.diagnostics} == set(result.covariates_used)
    assert result.confidence_interval_lower < result.adjusted_hazard_ratio < result.confidence_interval_upper
    assert apply_adjusted_cox_fdr((result,))[0].fdr == result.p_value
    assert normalize_stage("Stage IIIA") == "III"


def test_adjusted_cox_rejects_singular_expression() -> None:
    samples = tuple(item.model_copy(update={"expression": 1.0}) for item in _samples())
    with pytest.raises(ValueError, match="no variance"):
        run_adjusted_cox(AdjustedCoxRequest(project_id="TCGA-LUAD", gene="LNC1", samples=samples))


def test_adjusted_cox_reports_complete_case_missingness() -> None:
    samples = list(_samples())
    samples[0] = samples[0].model_copy(update={"age": None})
    samples[1] = samples[1].model_copy(update={"pathological_stage": None})
    result = run_adjusted_cox(AdjustedCoxRequest(project_id="TCGA-LUAD", gene="LNC1", samples=tuple(samples)))
    assert result.initial_survival_eligible_count == 80
    assert result.complete_case_count == 78
    assert result.excluded_missing_covariates == 2
    assert result.missingness_by_covariate["age"] == 1
    assert result.model_parameter_count == len(result.covariates_used)


def test_adjusted_cox_guards_events_per_parameter() -> None:
    with pytest.raises(ValueError, match="events/model complexity"):
        run_adjusted_cox(AdjustedCoxRequest(project_id="TCGA-LUAD", gene="LNC1", samples=_samples(), minimum_events_per_parameter=20))
