"""Deterministic count filtering, normalization, and differential expression."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import t as student_t

from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult


class ExpressionGroup(StrEnum):
    CASE = "case"
    CONTROL = "control"


class ExpressionDataType(StrEnum):
    RAW_INTEGER_COUNTS = "RAW_INTEGER_COUNTS"
    XENA_LOG2_STAR_COUNTS = "XENA_LOG2_STAR_COUNTS"
    NORMALIZED_CONTINUOUS = "NORMALIZED_CONTINUOUS"


class DifferentialExpressionMethod(StrEnum):
    AUTO = "AUTO"
    PYDESEQ2 = "PYDESEQ2"
    WELCH = "WELCH"


class ExpressionSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    sample_id: str = Field(min_length=1)
    group: ExpressionGroup


class CountFeature(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    feature_id: str = Field(min_length=1)
    gene_symbol: str | None = Field(default=None, min_length=1)
    counts: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts(self) -> CountFeature:
        if any(isinstance(value, bool) or value < 0 for value in self.counts):
            raise ValueError("counts must be non-negative integers")
        return self


class DifferentialExpressionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    project_id: str = Field(pattern=r"^TCGA-[A-Z0-9]+$")
    samples: tuple[ExpressionSample, ...] = Field(min_length=4)
    features: tuple[CountFeature, ...] = Field(min_length=1)
    case_definition: str = Field(default="Primary Tumor", min_length=1)
    control_definition: str = Field(default="Solid Tissue Normal", min_length=1)
    normalization: Literal["CPM then log2(CPM + pseudocount)"] = "CPM then log2(CPM + pseudocount)"
    test_method: Literal["two-sided Welch t-test"] = "two-sided Welch t-test"
    multiple_testing_method: Literal["Benjamini-Hochberg"] = "Benjamini-Hochberg"
    pseudocount: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    min_count: int = Field(default=10, ge=0)
    min_samples_per_group: int = Field(default=2, ge=1)
    min_case_samples: int = Field(default=20, ge=2)
    min_control_samples: int = Field(default=10, ge=2)
    alpha: float = Field(default=0.05, gt=0, lt=1, allow_inf_nan=False)
    candidate_genes: tuple[str, ...] = ()
    count_source: str = Field(default="GDC STAR counts", min_length=1)
    annotation_source: str = Field(default="source matrix annotation", min_length=1)
    method: DifferentialExpressionMethod = DifferentialExpressionMethod.WELCH
    input_data_type: ExpressionDataType = ExpressionDataType.RAW_INTEGER_COUNTS
    expression_transform: str = "none"
    expression_source_dataset: str | None = None
    upstream_source: str | None = None
    distribution_source: str | None = None

    @model_validator(mode="after")
    def validate_matrix(self) -> DifferentialExpressionRequest:
        if self.method == DifferentialExpressionMethod.PYDESEQ2 and self.input_data_type != ExpressionDataType.RAW_INTEGER_COUNTS:
            raise ValueError("PYDESEQ2 requires explicitly declared raw integer counts")
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample IDs must be unique")
        case_count = sum(sample.group == ExpressionGroup.CASE for sample in self.samples)
        control_count = len(self.samples) - case_count
        if case_count < self.min_case_samples:
            raise ValueError(f"differential expression requires at least {self.min_case_samples} case samples")
        if control_count < self.min_control_samples:
            raise ValueError(
                f"differential expression requires at least {self.min_control_samples} control samples"
            )
        if self.min_samples_per_group > min(case_count, control_count):
            raise ValueError("min_samples_per_group exceeds the smaller group size")
        feature_ids = [feature.feature_id for feature in self.features]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("feature IDs must be unique")
        if any(len(feature.counts) != len(self.samples) for feature in self.features):
            raise ValueError("every count vector must align with the sample list")
        library_sizes = [sum(feature.counts[index] for feature in self.features) for index in range(len(self.samples))]
        if any(size <= 0 for size in library_sizes):
            raise ValueError("every sample must have a positive library size")
        if len(self.candidate_genes) != len(set(self.candidate_genes)):
            raise ValueError("candidate_genes must be unique")
        known = {
            name
            for feature in self.features
            for name in (feature.feature_id, feature.gene_symbol)
            if name is not None
        }
        unknown = set(self.candidate_genes) - known
        if unknown:
            raise ValueError(f"candidate genes are absent from the matrix: {sorted(unknown)}")
        return self


class DifferentialExpressionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_definition: str
    control_definition: str
    normalization: str
    fold_change_definition: str
    test_method: str
    multiple_testing_method: str
    multiple_testing_scope: str
    pseudocount: float
    min_count: int
    min_samples_per_group: int
    min_case_samples: int
    min_control_samples: int
    filter_rule: str
    alpha: float
    method: str = DifferentialExpressionMethod.WELCH.value
    requested_method: str = DifferentialExpressionMethod.AUTO.value
    input_data_type: str = ExpressionDataType.RAW_INTEGER_COUNTS.value
    expression_transform: str = "none"
    expression_source_dataset: str | None = None
    upstream_source: str | None = None
    distribution_source: str | None = None
    genes_tested: int = 0
    lncrnas_tested: int = 0


class DifferentialExpressionFeatureResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_id: str
    gene_symbol: str | None = None
    case_mean_cpm: float = Field(ge=0, allow_inf_nan=False)
    control_mean_cpm: float = Field(ge=0, allow_inf_nan=False)
    log2_fold_change: float = Field(allow_inf_nan=False)
    test_statistic: float | None = Field(default=None, allow_inf_nan=False)
    degrees_of_freedom: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    p_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    adjusted_p_value: float = Field(ge=0, le=1, allow_inf_nan=False)
    significant: bool


class FilteredFeature(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_id: str
    gene_symbol: str | None = None
    reason: str


class DifferentialExpressionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    case_sample_count: int = Field(ge=2)
    control_sample_count: int = Field(ge=2)
    input_feature_count: int = Field(ge=1)
    selected_feature_count: int = Field(ge=0)
    tested_feature_count: int = Field(ge=0)
    filtered_feature_count: int = Field(ge=0)
    results: tuple[DifferentialExpressionFeatureResult, ...]
    filtered_features: tuple[FilteredFeature, ...]
    configuration: DifferentialExpressionConfig


def _welch_test(case: np.ndarray, control: np.ndarray) -> tuple[float | None, float | None, float]:
    case_mean = float(np.mean(case))
    control_mean = float(np.mean(control))
    case_variance = float(np.var(case, ddof=1))
    control_variance = float(np.var(control, ddof=1))
    variance_sum = case_variance / len(case) + control_variance / len(control)
    if variance_sum <= 1e-15:
        return (0.0, None, 1.0) if math.isclose(case_mean, control_mean) else (None, None, 0.0)
    statistic = (case_mean - control_mean) / math.sqrt(variance_sum)
    numerator = variance_sum**2
    denominator = (
        (case_variance / len(case)) ** 2 / (len(case) - 1)
        + (control_variance / len(control)) ** 2 / (len(control) - 1)
    )
    degrees = numerator / denominator
    p_value = float(2 * student_t.sf(abs(statistic), degrees))
    return float(statistic), float(degrees), p_value


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank_index in range(len(p_values) - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, p_values[original_index] * len(p_values) / rank)
        adjusted[original_index] = min(1.0, running)
    return [float(value) for value in adjusted]


def run_differential_expression(
    request: DifferentialExpressionRequest,
    *,
    count_backend: Callable[[DifferentialExpressionRequest], ToolResult[DifferentialExpressionResult]] | None = None,
) -> ToolResult[DifferentialExpressionResult]:
    """Route declared expression data to a statistically compatible DE backend."""

    selected_method = _select_method(request)
    if selected_method == DifferentialExpressionMethod.PYDESEQ2:
        backend = count_backend or _run_pydeseq2
        return backend(request)
    return _run_welch(request, selected_method=selected_method)


def _select_method(request: DifferentialExpressionRequest) -> DifferentialExpressionMethod:
    if request.method != DifferentialExpressionMethod.AUTO:
        return request.method
    if request.input_data_type == ExpressionDataType.RAW_INTEGER_COUNTS:
        return DifferentialExpressionMethod.PYDESEQ2
    return DifferentialExpressionMethod.WELCH


def _run_welch(request: DifferentialExpressionRequest, *, selected_method: DifferentialExpressionMethod = DifferentialExpressionMethod.WELCH) -> ToolResult[DifferentialExpressionResult]:
    """Run the existing CPM/log2 normalization and two-sided Welch tests."""

    case_mask = np.asarray([sample.group == ExpressionGroup.CASE for sample in request.samples])
    control_mask = ~case_mask
    matrix = np.asarray([feature.counts for feature in request.features], dtype=float)
    library_sizes = np.sum(matrix, axis=0)
    cpm = matrix / library_sizes * 1_000_000.0
    log_cpm = np.log2(cpm + request.pseudocount)
    candidates = set(request.candidate_genes)
    selected = [
        (index, feature)
        for index, feature in enumerate(request.features)
        if not candidates or feature.feature_id in candidates or feature.gene_symbol in candidates
    ]
    tested: list[tuple[CountFeature, float, float, float, float | None, float | None, float]] = []
    filtered: list[FilteredFeature] = []
    for index, feature in selected:
        raw = matrix[index]
        case_expressed = int(np.sum(raw[case_mask] >= request.min_count))
        control_expressed = int(np.sum(raw[control_mask] >= request.min_count))
        if max(case_expressed, control_expressed) < request.min_samples_per_group:
            filtered.append(
                FilteredFeature(
                    feature_id=feature.feature_id,
                    gene_symbol=feature.gene_symbol,
                    reason=(
                        f"count >= {request.min_count} in fewer than "
                        f"{request.min_samples_per_group} samples in either group"
                    ),
                )
            )
            continue
        case_mean_cpm = float(np.mean(cpm[index, case_mask]))
        control_mean_cpm = float(np.mean(cpm[index, control_mask]))
        log2_fc = math.log2(
            (case_mean_cpm + request.pseudocount)
            / (control_mean_cpm + request.pseudocount)
        )
        statistic, degrees, p_value = _welch_test(
            log_cpm[index, case_mask], log_cpm[index, control_mask]
        )
        tested.append(
            (feature, case_mean_cpm, control_mean_cpm, log2_fc, statistic, degrees, p_value)
        )
    adjusted = _benjamini_hochberg([item[-1] for item in tested])
    results = tuple(
        DifferentialExpressionFeatureResult(
            feature_id=feature.feature_id,
            gene_symbol=feature.gene_symbol,
            case_mean_cpm=case_mean,
            control_mean_cpm=control_mean,
            log2_fold_change=log2_fc,
            test_statistic=statistic,
            degrees_of_freedom=degrees,
            p_value=p_value,
            adjusted_p_value=adjusted_p,
            significant=adjusted_p <= request.alpha,
        )
        for (feature, case_mean, control_mean, log2_fc, statistic, degrees, p_value), adjusted_p
        in zip(tested, adjusted)
    )
    scope = "candidate subset" if candidates else "all selected matrix features"
    data = DifferentialExpressionResult(
        project_id=request.project_id,
        case_sample_count=int(np.sum(case_mask)),
        control_sample_count=int(np.sum(control_mask)),
        input_feature_count=len(request.features),
        selected_feature_count=len(selected),
        tested_feature_count=len(results),
        filtered_feature_count=len(filtered),
        results=results,
        filtered_features=tuple(filtered),
        configuration=DifferentialExpressionConfig(
            case_definition=request.case_definition,
            control_definition=request.control_definition,
            normalization=request.normalization,
            fold_change_definition="log2((mean case CPM + pseudocount) / (mean control CPM + pseudocount))",
            test_method=request.test_method,
            multiple_testing_method=request.multiple_testing_method,
            multiple_testing_scope=scope,
            pseudocount=request.pseudocount,
            min_count=request.min_count,
            min_samples_per_group=request.min_samples_per_group,
            min_case_samples=request.min_case_samples,
            min_control_samples=request.min_control_samples,
            filter_rule="retain when count threshold is met in enough samples in either group",
            alpha=request.alpha,
            method=selected_method.value,
            requested_method=request.method.value,
            input_data_type=request.input_data_type.value,
            expression_transform=request.expression_transform,
            expression_source_dataset=request.expression_source_dataset,
            upstream_source=request.upstream_source,
            distribution_source=request.distribution_source,
            genes_tested=len(results),
            lncrnas_tested=len(results),
        ),
    )
    warnings = []
    if filtered:
        warnings.append(f"Filtered {len(filtered)} low-count feature(s) before testing.")
    if not results:
        warnings.append("No features passed the configured count filter.")
    return ToolResult[DifferentialExpressionResult](
        status=ResultStatus.SUCCESS,
        data=data,
        warnings=warnings,
        provenance=[
            Provenance(source=request.count_source, source_id=request.project_id),
            Provenance(source=request.annotation_source),
        ],
    )


def _run_pydeseq2(request: DifferentialExpressionRequest) -> ToolResult[DifferentialExpressionResult]:
    """Run PyDESeq2 lazily for explicitly declared raw integer count matrices.

    PyDESeq2 is an optional statistics dependency because the deployed Xena matrix is
    transformed and correctly routes to Welch. Raw-count users receive a clear error
    rather than an inappropriate silent fallback.
    """
    try:
        import pandas as pd
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:
        return ToolResult(
            status=ResultStatus.FAILURE,
            error=ToolError(code="count_de_backend_unavailable", message="PyDESeq2 is required for RAW_INTEGER_COUNTS; install oncolncai[statistics]", retryable=False),
            provenance=[Provenance(source=request.count_source, source_id=request.project_id)],
        )
    selected = [feature for feature in request.features if not request.candidate_genes or feature.feature_id in request.candidate_genes or feature.gene_symbol in request.candidate_genes]
    counts = pd.DataFrame({feature.feature_id: feature.counts for feature in selected}, index=[sample.sample_id for sample in request.samples])
    metadata = pd.DataFrame({"condition": ["case" if sample.group == ExpressionGroup.CASE else "control" for sample in request.samples]}, index=counts.index)
    keep = (counts >= request.min_count).groupby(metadata["condition"]).sum().max(axis=0) >= request.min_samples_per_group
    filtered_ids = list(counts.columns[~keep])
    counts = counts.loc[:, keep]
    if counts.empty:
        return ToolResult(status=ResultStatus.FAILURE, error=ToolError(code="no_features_after_filtering", message="No features passed raw-count filtering", retryable=False))
    dds = DeseqDataSet(counts=counts, metadata=metadata, design="~condition", refit_cooks=True, n_cpus=1)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["condition", "case", "control"], n_cpus=1)
    stats.summary()
    table = stats.results_df
    case_mask = np.asarray([sample.group == ExpressionGroup.CASE for sample in request.samples])
    matrix = np.asarray([feature.counts for feature in selected], dtype=float)
    library_sizes = np.sum(matrix, axis=0)
    cpm = matrix / library_sizes * 1_000_000.0
    by_id = {feature.feature_id: (index, feature) for index, feature in enumerate(selected)}
    results = []
    for feature_id, row in table.iterrows():
        if not all(math.isfinite(float(row[key])) for key in ("log2FoldChange", "pvalue", "padj")):
            continue
        index, feature = by_id[str(feature_id)]
        results.append(DifferentialExpressionFeatureResult(
            feature_id=feature.feature_id, gene_symbol=feature.gene_symbol,
            case_mean_cpm=float(np.mean(cpm[index, case_mask])), control_mean_cpm=float(np.mean(cpm[index, ~case_mask])),
            log2_fold_change=float(row["log2FoldChange"]), p_value=float(row["pvalue"]),
            adjusted_p_value=float(row["padj"]), significant=float(row["padj"]) <= request.alpha,
        ))
    config = DifferentialExpressionConfig(
        case_definition=request.case_definition, control_definition=request.control_definition,
        normalization="PyDESeq2 median-of-ratios size factors; negative-binomial GLM",
        fold_change_definition="PyDESeq2 condition case vs control coefficient", test_method="PyDESeq2 Wald test",
        multiple_testing_method="Benjamini-Hochberg", multiple_testing_scope="all selected matrix features",
        pseudocount=request.pseudocount, min_count=request.min_count, min_samples_per_group=request.min_samples_per_group,
        min_case_samples=request.min_case_samples, min_control_samples=request.min_control_samples,
        filter_rule="raw count threshold in enough samples in either group", alpha=request.alpha,
        method=DifferentialExpressionMethod.PYDESEQ2.value, requested_method=request.method.value,
        input_data_type=request.input_data_type.value, expression_transform=request.expression_transform,
        expression_source_dataset=request.expression_source_dataset, upstream_source=request.upstream_source,
        distribution_source=request.distribution_source, genes_tested=len(results), lncrnas_tested=len(results),
    )
    return ToolResult(status=ResultStatus.SUCCESS, data=DifferentialExpressionResult(
        project_id=request.project_id, case_sample_count=int(np.sum(case_mask)), control_sample_count=int(np.sum(~case_mask)),
        input_feature_count=len(request.features), selected_feature_count=len(selected), tested_feature_count=len(results),
        filtered_feature_count=len(filtered_ids), results=tuple(results),
        filtered_features=tuple(FilteredFeature(feature_id=item, reason="raw-count filter") for item in filtered_ids), configuration=config,
    ), provenance=[Provenance(source=request.count_source, source_id=request.project_id), Provenance(source=request.annotation_source)])
