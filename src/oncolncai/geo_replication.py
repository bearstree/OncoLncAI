"""Deterministic GEO expression replication after suitability assessment."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import ttest_ind

from oncolncai.differential_expression import _benjamini_hochberg


class GEOExpressionSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    group: str = Field(pattern=r"^(tumor|normal)$")


class GEOExpressionFeature(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    platform_identifier: str
    gene_symbol: str | None = None
    ensembl_gene_id: str | None = None
    values: tuple[float, ...]


class GEOReplicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    accession: str = Field(pattern=r"^GSE\d+$")
    platform: str = Field(pattern=r"^GPL\d+$")
    cancer: str
    samples: tuple[GEOExpressionSample, ...]
    features: tuple[GEOExpressionFeature, ...]
    candidate_symbols: tuple[str, ...]
    tcga_directions: dict[str, float]
    minimum_group_size: int = Field(default=3, ge=2)

    @model_validator(mode="after")
    def validate_matrix(self) -> "GEOReplicationRequest":
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("GEO sample IDs must be unique")
        if any(len(item.values) != len(self.samples) for item in self.features):
            raise ValueError("GEO feature vectors must align with samples")
        return self


class GEOReplicationFeatureResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gene_symbol: str
    platform_identifiers: tuple[str, ...]
    effect_estimate: float
    p_value: float = Field(ge=0, le=1)
    fdr: float = Field(ge=0, le=1)
    direction_consistent_with_tcga: bool


class GEOReplicationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    accession: str
    platform: str
    tumor_count: int
    normal_count: int
    mapped_candidates: tuple[str, ...]
    unmapped_candidates: tuple[str, ...]
    results: tuple[GEOReplicationFeatureResult, ...]
    analysis: str = "GEO expression replication"


def run_geo_expression_replication(request: GEOReplicationRequest) -> GEOReplicationResult:
    tumor = np.asarray([item.group == "tumor" for item in request.samples])
    normal = ~tumor
    if int(np.sum(tumor)) < request.minimum_group_size or int(np.sum(normal)) < request.minimum_group_size:
        raise ValueError("GEO cohort is not eligible: inadequate tumor/normal sample count")
    by_symbol: dict[str, list[GEOExpressionFeature]] = {}
    for feature in request.features:
        if feature.gene_symbol:
            by_symbol.setdefault(feature.gene_symbol.upper(), []).append(feature)
    raw: list[tuple[str, tuple[str, ...], float, float]] = []
    unmapped: list[str] = []
    for candidate in request.candidate_symbols:
        probes = by_symbol.get(candidate.upper(), [])
        if not probes:
            unmapped.append(candidate)
            continue
        values = np.mean(np.asarray([probe.values for probe in probes], dtype=float), axis=0)
        effect = float(np.mean(values[tumor]) - np.mean(values[normal]))
        test = ttest_ind(values[tumor], values[normal], equal_var=False)
        p_value = float(test.pvalue) if math.isfinite(float(test.pvalue)) else 1.0
        raw.append((candidate, tuple(probe.platform_identifier for probe in probes), effect, p_value))
    fdrs = _benjamini_hochberg([item[3] for item in raw])
    results = tuple(GEOReplicationFeatureResult(
        gene_symbol=gene, platform_identifiers=probes, effect_estimate=effect, p_value=p_value, fdr=fdr,
        direction_consistent_with_tcga=(effect == 0 and request.tcga_directions.get(gene, 0) == 0) or effect * request.tcga_directions.get(gene, 0) > 0,
    ) for (gene, probes, effect, p_value), fdr in zip(raw, fdrs))
    return GEOReplicationResult(
        accession=request.accession, platform=request.platform,
        tumor_count=int(np.sum(tumor)), normal_count=int(np.sum(normal)),
        mapped_candidates=tuple(item[0] for item in raw), unmapped_candidates=tuple(unmapped), results=results,
    )
