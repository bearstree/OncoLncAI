import math

import pytest
from pydantic import ValidationError

from oncolncai import (
    CountFeature,
    DifferentialExpressionRequest,
    ExpressionSample,
    ResultStatus,
    ToolName,
    build_default_registry,
    run_differential_expression,
)
from oncolncai.differential_expression import DifferentialExpressionMethod, ExpressionDataType


def known_behavior_request(**changes) -> DifferentialExpressionRequest:
    samples = tuple(
        ExpressionSample(sample_id=f"case-{i}", group="case") for i in range(4)
    ) + tuple(
        ExpressionSample(sample_id=f"control-{i}", group="control") for i in range(4)
    )
    up = (100, 110, 90, 105, 10, 12, 8, 11)
    down = (10, 12, 8, 11, 100, 110, 90, 105)
    null = (50, 52, 48, 51, 50, 52, 48, 51)
    low = (0, 1, 0, 1, 0, 1, 0, 1)
    background = tuple(1000 - sum(values) for values in zip(up, down, null, low))
    values = {
        "project_id": "TCGA-LUAD",
        "samples": samples,
        "features": (
            CountFeature(feature_id="ENSG-UP", gene_symbol="UP", counts=up),
            CountFeature(feature_id="ENSG-DOWN", gene_symbol="DOWN", counts=down),
            CountFeature(feature_id="ENSG-NULL", gene_symbol="NULL", counts=null),
            CountFeature(feature_id="ENSG-LOW", gene_symbol="LOW", counts=low),
            CountFeature(feature_id="ENSG-BG", gene_symbol="BACKGROUND", counts=background),
        ),
        "count_source": "synthetic count fixture",
        "annotation_source": "synthetic annotation fixture",
        "min_case_samples": 4,
        "min_control_samples": 4,
    }
    values.update(changes)
    return DifferentialExpressionRequest(**values)


def result_by_symbol(result):
    return {item.gene_symbol: item for item in result.data.results}


def test_known_up_down_and_null_features_have_expected_behavior() -> None:
    result = run_differential_expression(known_behavior_request())

    assert result.status == ResultStatus.SUCCESS
    assert result.data is not None
    rows = result_by_symbol(result)
    assert rows["UP"].log2_fold_change > 3
    assert rows["DOWN"].log2_fold_change < -3
    assert rows["UP"].adjusted_p_value < 0.01
    assert rows["DOWN"].adjusted_p_value < 0.01
    assert rows["NULL"].log2_fold_change == pytest.approx(0, abs=0.02)
    assert rows["NULL"].adjusted_p_value > 0.5
    assert rows["UP"].significant and rows["DOWN"].significant


def test_low_counts_are_filtered_before_bh_correction() -> None:
    result = run_differential_expression(known_behavior_request())

    assert result.data.tested_feature_count == 4
    assert result.data.filtered_feature_count == 1
    assert result.data.filtered_features[0].gene_symbol == "LOW"
    assert "Filtered 1 low-count" in result.warnings[0]
    p_values = [item.p_value for item in result.data.results]
    adjusted = [item.adjusted_p_value for item in result.data.results]
    assert all(adjusted_p >= p for p, adjusted_p in zip(p_values, adjusted))
    assert all(0 <= value <= 1 for value in adjusted)


def test_candidate_filter_defines_multiple_testing_scope() -> None:
    result = run_differential_expression(
        known_behavior_request(candidate_genes=("UP", "NULL"))
    )

    assert result.data.selected_feature_count == 2
    assert result.data.tested_feature_count == 2
    assert {item.gene_symbol for item in result.data.results} == {"UP", "NULL"}
    assert result.data.configuration.multiple_testing_scope == "candidate subset"


def test_counts_configuration_and_provenance_are_preserved() -> None:
    result = run_differential_expression(known_behavior_request())

    assert result.data.case_sample_count == 4
    assert result.data.control_sample_count == 4
    assert result.data.configuration.test_method == "two-sided Welch t-test"
    assert result.data.configuration.multiple_testing_method == "Benjamini-Hochberg"
    assert result.data.configuration.pseudocount == 1
    assert [(item.source, item.source_id) for item in result.provenance] == [
        ("synthetic count fixture", "TCGA-LUAD"),
        ("synthetic annotation fixture", None),
    ]


def test_all_filtered_returns_success_with_explicit_warning() -> None:
    request = known_behavior_request(
        features=(
            CountFeature(
                feature_id="ENSG-LOW",
                gene_symbol="LOW",
                counts=(1, 1, 1, 1, 1, 1, 1, 1),
            ),
            CountFeature(
                feature_id="ENSG-BG",
                gene_symbol="BACKGROUND",
                counts=(999, 999, 999, 999, 999, 999, 999, 999),
            ),
        ),
        candidate_genes=("LOW",),
    )
    result = run_differential_expression(request)

    assert result.status == ResultStatus.SUCCESS
    assert result.data.tested_feature_count == 0
    assert "No features passed" in result.warnings[-1]


def test_invalid_matrices_are_rejected_before_analysis() -> None:
    base = known_behavior_request()
    with pytest.raises(ValidationError, match="align"):
        DifferentialExpressionRequest.model_validate(
            {
                **base.model_dump(),
                "features": [
                    {"feature_id": "short", "counts": [1, 2, 3]},
                ],
            }
        )
    with pytest.raises(ValidationError, match="positive library"):
        DifferentialExpressionRequest(
            project_id="TCGA-LUAD",
            samples=base.samples,
            features=(CountFeature(feature_id="zero", counts=(0,) * 8),),
            min_case_samples=4,
            min_control_samples=4,
        )
    with pytest.raises(ValidationError):
        CountFeature(feature_id="bad", counts=(1, -1, 2, 3))
    with pytest.raises(ValidationError, match="absent"):
        known_behavior_request(candidate_genes=("NOT-IN-MATRIX",))
    with pytest.raises(ValidationError, match="at least 20 case"):
        DifferentialExpressionRequest(
            project_id="TCGA-LUAD",
            samples=base.samples,
            features=base.features,
        )


def test_no_nonfinite_statistics_escape() -> None:
    result = run_differential_expression(known_behavior_request())
    for item in result.data.results:
        values = [
            item.case_mean_cpm,
            item.control_mean_cpm,
            item.log2_fold_change,
            item.p_value,
            item.adjusted_p_value,
        ]
        assert all(math.isfinite(value) for value in values)


def test_de_tool_is_registered_and_executes_typed_input() -> None:
    request = known_behavior_request()
    result = build_default_registry().execute(
        ToolName.RUN_DIFFERENTIAL_EXPRESSION, request.model_dump(mode="json")
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.data.tested_feature_count == 4


def test_auto_routes_declared_continuous_data_to_welch() -> None:
    request = known_behavior_request(method=DifferentialExpressionMethod.AUTO,
                                     input_data_type=ExpressionDataType.XENA_LOG2_STAR_COUNTS,
                                     expression_transform="log2(count+1)")
    result = run_differential_expression(request)
    assert result.data.configuration.method == "WELCH"
    assert result.data.configuration.input_data_type == "XENA_LOG2_STAR_COUNTS"
    assert result.data.configuration.genes_tested == result.data.tested_feature_count


def test_auto_routes_raw_counts_to_injected_count_backend() -> None:
    request = known_behavior_request(method=DifferentialExpressionMethod.AUTO,
                                     input_data_type=ExpressionDataType.RAW_INTEGER_COUNTS)
    expected = run_differential_expression(known_behavior_request())
    calls = []
    result = run_differential_expression(request, count_backend=lambda value: calls.append(value) or expected)
    assert calls == [request]
    assert result is expected


def test_count_model_rejects_normalized_continuous_input() -> None:
    with pytest.raises(ValidationError, match="raw integer counts"):
        known_behavior_request(method=DifferentialExpressionMethod.PYDESEQ2,
                               input_data_type=ExpressionDataType.NORMALIZED_CONTINUOUS)


def test_installed_pydeseq2_backend_returns_common_schema() -> None:
    pytest.importorskip("pydeseq2")
    result = run_differential_expression(known_behavior_request(
        method=DifferentialExpressionMethod.AUTO,
        input_data_type=ExpressionDataType.RAW_INTEGER_COUNTS,
    ))
    assert result.status == ResultStatus.SUCCESS
    assert result.data.configuration.method == "PYDESEQ2"
    assert result.data.configuration.input_data_type == "RAW_INTEGER_COUNTS"
    assert result.data.case_sample_count == 4
