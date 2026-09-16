import pytest
from pydantic import ValidationError

from oncolncai import (
    ConsistencyLabel,
    DifferentialExpressionEvidenceInput,
    EvidenceDirection,
    EvidenceFusionConfig,
    EvidenceFusionRequest,
    GEOValidationEvidenceInput,
    LiteratureEvidenceInput,
    MockLLMProvider,
    Provenance,
    SurvivalEvidenceInput,
    ToolName,
    build_default_registry,
    prioritize_evidence,
    synthesize_evidence_prioritization,
)


TCGA_PROVENANCE = (Provenance(source="GDC STAR counts", source_id="TCGA-LUAD"),)
SURVIVAL_PROVENANCE = (Provenance(source="TCGA PanCanAtlas", source_id="TCGA-LUAD"),)
GEO_PROVENANCE = (Provenance(source="NCBI GEO", source_id="GSE100001"),)
LITERATURE_PROVENANCE = (Provenance(source="NCBI PubMed", source_id="1001", query="LUAD lncRNA"),)


def complete_request(**changes) -> EvidenceFusionRequest:
    values = {
        "candidate": "MALAT1",
        "cancer": "lung adenocarcinoma",
        "literature": (
            LiteratureEvidenceInput(pmid="1001", provenance=LITERATURE_PROVENANCE, expression_direction=EvidenceDirection.UP, prognostic_direction=EvidenceDirection.ADVERSE),
        ),
        "tcga": DifferentialExpressionEvidenceInput(project_id="TCGA-LUAD", log2_fold_change=1.0, adjusted_p_value=0.01, provenance=TCGA_PROVENANCE),
        "survival": SurvivalEvidenceInput(project_id="TCGA-LUAD", hazard_ratio=2.0, cox_p_value=0.01, log_rank_p_value=0.02, provenance=SURVIVAL_PROVENANCE),
        "geo": GEOValidationEvidenceInput(accession="GSE100001", log2_fold_change=1.0, adjusted_p_value=0.01, provenance=GEO_PROVENANCE),
    }
    values.update(changes)
    return EvidenceFusionRequest(**values)


def score(request: EvidenceFusionRequest) -> float:
    result = prioritize_evidence(request)
    assert result.data is not None
    return result.data.evidence_prioritization_score


def test_component_scores_and_manual_weighted_score_are_exposed() -> None:
    result = prioritize_evidence(complete_request())
    assert result.data is not None
    components = {item.component: item for item in result.data.components}
    assert components["literature"].score == pytest.approx(1 / 3)
    assert components["tcga_differential_expression"].score == 1.0
    assert components["survival"].score == 1.0
    assert components["geo_replication"].score == 1.0
    assert components["cross_source_consistency"].score == 1.0
    expected = 100 * ((1 / 3) * 0.25 + 1 * 0.25 + 1 * 0.20 + 1 * 0.20 + 1 * 0.10)
    assert result.data.evidence_prioritization_score == pytest.approx(expected)
    assert result.data.consistency.label == ConsistencyLabel.CONSISTENT


def test_predictable_score_decrease_when_each_statistical_component_weakens() -> None:
    baseline = score(complete_request())
    weak_tcga = score(complete_request(tcga=DifferentialExpressionEvidenceInput(project_id="TCGA-LUAD", log2_fold_change=1, adjusted_p_value=0.2, provenance=TCGA_PROVENANCE)))
    weak_survival = score(complete_request(survival=SurvivalEvidenceInput(project_id="TCGA-LUAD", hazard_ratio=2, cox_p_value=0.2, log_rank_p_value=0.01, provenance=SURVIVAL_PROVENANCE)))
    weak_geo = score(complete_request(geo=GEOValidationEvidenceInput(accession="GSE100001", log2_fold_change=1, adjusted_p_value=0.2, provenance=GEO_PROVENANCE)))
    assert baseline > weak_tcga
    assert baseline > weak_survival
    assert baseline > weak_geo


def test_higher_literature_support_predictably_increases_score() -> None:
    low = score(complete_request(literature=(LiteratureEvidenceInput(pmid="1001", provenance=LITERATURE_PROVENANCE),)))
    high = score(complete_request(literature=(LiteratureEvidenceInput(pmid="1001", provenance=LITERATURE_PROVENANCE), LiteratureEvidenceInput(pmid="1002", provenance=(Provenance(source="NCBI PubMed", source_id="1002"),)))))
    assert high > low


def test_missing_evidence_is_excluded_not_scored_as_zero() -> None:
    config = EvidenceFusionConfig(consistency_weight=0)
    literature = (LiteratureEvidenceInput(pmid="1001", provenance=LITERATURE_PROVENANCE), LiteratureEvidenceInput(pmid="1002", provenance=(Provenance(source="NCBI PubMed", source_id="1002"),)), LiteratureEvidenceInput(pmid="1003", provenance=(Provenance(source="NCBI PubMed", source_id="1003"),)))
    result = prioritize_evidence(complete_request(literature=literature, tcga=None, survival=None, geo=None, config=config))
    assert result.data is not None
    assert result.data.evidence_prioritization_score == pytest.approx(100)
    assert result.data.evidence_coverage == pytest.approx(0.25 / 0.9)
    assert set(result.data.missing_evidence) == {"tcga_differential_expression", "survival", "geo_replication", "cross_source_consistency"}


def test_contradiction_is_visible_and_reduces_score() -> None:
    consistent = prioritize_evidence(complete_request())
    contradictory = prioritize_evidence(complete_request(geo=GEOValidationEvidenceInput(accession="GSE100001", log2_fold_change=-1, adjusted_p_value=0.01, provenance=GEO_PROVENANCE)))
    assert consistent.data is not None and contradictory.data is not None
    assert contradictory.data.consistency.contradictions == 2
    assert contradictory.data.consistency.label == ConsistencyLabel.MIXED
    assert contradictory.data.evidence_prioritization_score < consistent.data.evidence_prioritization_score


def test_weights_are_configurable() -> None:
    request = complete_request(
        tcga=DifferentialExpressionEvidenceInput(project_id="TCGA-LUAD", log2_fold_change=1, adjusted_p_value=0.2, provenance=TCGA_PROVENANCE),
        config=EvidenceFusionConfig(literature_weight=0, tcga_weight=1, survival_weight=0, geo_weight=0, consistency_weight=0),
    )
    assert score(request) == 0


def test_synthesis_can_explain_but_schema_cannot_change_score() -> None:
    numeric = prioritize_evidence(complete_request()).data
    provider = MockLLMProvider([{"explanation": "The deterministic components are strong.", "limitations": ["Research prioritization only."]}])
    synthesis = synthesize_evidence_prioritization(numeric, provider=provider)
    assert synthesis.explanation.startswith("The deterministic")
    assert "evidence_prioritization_score" in provider.calls[0][0]
    bad_provider = MockLLMProvider([{"explanation": "Changed", "limitations": [], "score": 100}])
    with pytest.raises(ValidationError):
        synthesize_evidence_prioritization(numeric, provider=bad_provider)


def test_duplicate_pmids_and_empty_literature_are_rejected() -> None:
    item = LiteratureEvidenceInput(pmid="1001", provenance=LITERATURE_PROVENANCE)
    with pytest.raises(ValidationError, match="unique"):
        complete_request(literature=(item, item))
    with pytest.raises(ValidationError, match="null"):
        complete_request(literature=())
    with pytest.raises(ValidationError, match="at least one"):
        EvidenceFusionRequest(candidate="MALAT1", cancer="LUAD")


def test_registry_dispatches_typed_prioritization_tool() -> None:
    payload = complete_request().model_dump(mode="json")
    result = build_default_registry().execute(ToolName.PRIORITIZE_EVIDENCE, payload)
    assert result.data is not None
    assert result.data.evidence_prioritization_score == pytest.approx(83.333333)
