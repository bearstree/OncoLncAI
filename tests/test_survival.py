import math

import pytest
from pydantic import ValidationError

from oncolncai import (
    ResultStatus,
    SurvivalRequest,
    SurvivalSample,
    ToolName,
    build_default_registry,
    run_survival_analysis,
)


def paired_null_samples() -> tuple[SurvivalSample, ...]:
    samples = []
    for index in range(20):
        time = float(index + 1)
        event = int(index % 2 == 0)
        samples.append(
            SurvivalSample(sample_id=f"low-{index}", time=time, event=event, expression=float(index))
        )
        samples.append(
            SurvivalSample(sample_id=f"high-{index}", time=time, event=event, expression=float(index + 20))
        )
    return tuple(samples)


def adverse_samples() -> tuple[SurvivalSample, ...]:
    low_times = (8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46)
    high_times = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    return tuple(
        SurvivalSample(
            sample_id=f"low-{index}", time=float(time), event=1,
            expression=float(index),
        )
        for index, time in enumerate(low_times)
    ) + tuple(
        SurvivalSample(
            sample_id=f"high-{index}", time=float(time), event=1,
            expression=float(index + 20),
        )
        for index, time in enumerate(high_times)
    )


def test_identical_group_outcomes_have_null_statistics_and_km_curves() -> None:
    result = run_survival_analysis(
        SurvivalRequest(project_id="TCGA-LUAD", gene="NEAT1", samples=paired_null_samples())
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.sample_count == 40
    assert result.data.event_count == 20
    assert result.data.hazard_ratio == pytest.approx(1.0, abs=1e-10)
    assert result.data.cox_p_value == pytest.approx(1.0, abs=1e-10)
    assert result.data.log_rank_p_value == pytest.approx(1.0, abs=1e-10)
    assert result.data.groups[0].curve == result.data.groups[1].curve
    assert result.data.confidence_interval_lower < 1 < result.data.confidence_interval_upper
    assert result.data.diagnostics.proportional_hazards_test.startswith("Spearman")
    assert result.data.diagnostics.proportional_hazards_assumption_met in (True, False, None)


def test_earlier_high_group_events_produce_adverse_significant_association() -> None:
    result = run_survival_analysis(
        SurvivalRequest(project_id="TCGA-BRCA", gene="MALAT1", samples=adverse_samples())
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.hazard_ratio > 3
    assert result.data.confidence_interval_lower > 1
    assert result.data.cox_p_value < 0.01
    assert result.data.log_rank_p_value < 0.01
    assert result.data.diagnostics.converged is True
    assert result.data.diagnostics.proportional_hazards_p_value is not None


def test_configuration_counts_and_provenance_are_preserved() -> None:
    request = SurvivalRequest(
        project_id="TCGA-KIRC",
        gene="MALAT1",
        samples=paired_null_samples(),
        endpoint="DSS",
        cohort_source="fixture clinical table",
        expression_source="fixture expression matrix",
    )
    result = run_survival_analysis(request)

    assert result.data is not None
    assert result.data.configuration.endpoint.value == "DSS"
    assert result.data.configuration.alpha == 0.05
    assert result.data.configuration.high_group_rule == "expression > median; ties assigned to low"
    assert result.data.configuration.ties_method == "Breslow"
    assert [(item.source, item.source_id) for item in result.provenance] == [
        ("fixture clinical table", "TCGA-KIRC"),
        ("fixture expression matrix", "MALAT1"),
    ]


@pytest.mark.parametrize(
    "samples, message",
    [
        (
            tuple(
                SurvivalSample(sample_id=f"s-{i}", time=i + 1, event=int(i == 0), expression=i)
                for i in range(20)
            ),
            "at least 5 events",
        ),
        (
            tuple(
                SurvivalSample(sample_id=f"s-{i}", time=i + 1, event=int(i < 5), expression=1.0)
                for i in range(20)
            ),
            "empty expression group",
        ),
    ],
)
def test_request_rejects_insufficient_or_degenerate_inputs(samples, message) -> None:
    with pytest.raises(ValidationError, match=message):
        SurvivalRequest(project_id="TCGA-LUAD", gene="NEAT1", samples=samples)


def test_request_rejects_duplicate_ids_and_nonfinite_values() -> None:
    duplicate = list(paired_null_samples())
    duplicate[1] = duplicate[1].model_copy(update={"sample_id": duplicate[0].sample_id})
    with pytest.raises(ValidationError, match="unique"):
        SurvivalRequest(project_id="TCGA-LUAD", gene="NEAT1", samples=tuple(duplicate))
    with pytest.raises(ValidationError):
        SurvivalSample(sample_id="bad", time=1, event=1, expression=math.nan)
    with pytest.raises(ValidationError, match="0.05"):
        SurvivalRequest(
            project_id="TCGA-LUAD",
            gene="NEAT1",
            samples=paired_null_samples(),
            alpha=0.10,
        )


def test_survival_tool_is_registered_and_executes_typed_request() -> None:
    result = build_default_registry().execute(
        ToolName.RUN_SURVIVAL_ANALYSIS,
        {
            "project_id": "TCGA-LUAD",
            "gene": "NEAT1",
            "samples": [sample.model_dump() for sample in paired_null_samples()],
        },
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.data.hazard_ratio == pytest.approx(1.0)
