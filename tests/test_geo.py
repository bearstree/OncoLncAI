from datetime import datetime, timezone

from oncolncai import (
    CriterionStatus,
    GEOAnnotation,
    GEOAssay,
    GEOClient,
    GEOSearchRequest,
    GEOSeriesMetadata,
    MockLLMProvider,
    ResultStatus,
    ToolName,
    build_default_registry,
    build_geo_query,
)


NOW = datetime(2026, 9, 14, tzinfo=timezone.utc)


class FakeBackend:
    def __init__(self, records=(), error: Exception | None = None):
        self.records = tuple(records)
        self.error = error
        self.calls = []

    def search(self, query: str, max_results: int, timeout: float):
        self.calls.append((query, max_results, timeout))
        if self.error:
            raise self.error
        return self.records


def record(accession: str, **overrides) -> GEOSeriesMetadata:
    values = {
        "accession": accession,
        "title": "Lung adenocarcinoma RNA expression",
        "summary": "Primary lung adenocarcinoma and adjacent normal lung.",
        "organism": "Homo sapiens",
        "disease_terms": ("lung adenocarcinoma",),
        "tissue_terms": ("lung",),
        "assay": GEOAssay.RNA_SEQ,
        "platform_accessions": ("GPL24676",),
        "case_sample_count": 24,
        "control_sample_count": 12,
        "annotation": GEOAnnotation.ENSEMBL_GENE,
        "metadata_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
        "retrieved_at": NOW,
    }
    values.update(overrides)
    return GEOSeriesMetadata(**values)


def request(**overrides) -> GEOSearchRequest:
    values = {"cancer": "lung adenocarcinoma", "tissue": "lung"}
    values.update(overrides)
    return GEOSearchRequest(**values)


def test_search_filters_and_ranks_by_explicit_suitability() -> None:
    suitable = record("GSE100001")
    too_small = record("GSE100002", case_sample_count=4, control_sample_count=3)
    wrong_tissue = record("GSE100003", tissue_terms=("blood",))
    backend = FakeBackend((too_small, suitable, wrong_tissue))

    result = GEOClient(backend=backend).search(request())

    assert result.status == ResultStatus.SUCCESS
    assert result.data is not None
    assert [item.dataset.accession for item in result.data.suitable_datasets] == ["GSE100001"]
    assert not result.data.no_suitable_validation_dataset
    assert [item.source_id for item in result.provenance] == ["GSE100002", "GSE100001", "GSE100003"]
    assert all(item.query == result.data.query for item in result.provenance)
    failed = {item.dataset.accession: item for item in result.data.assessments}
    assert next(c for c in failed["GSE100002"].criteria if c.criterion == "sample_size").status == CriterionStatus.FAIL
    assert next(c for c in failed["GSE100003"].criteria if c.criterion == "tissue_match").status == CriterionStatus.FAIL


def test_returns_no_suitable_dataset_without_forcing_validation() -> None:
    unsuitable = record(
        "GSE200001",
        assay=GEOAssay.OTHER,
        case_sample_count=18,
        control_sample_count=0,
        annotation=GEOAnnotation.UNKNOWN,
    )

    result = GEOClient(backend=FakeBackend((unsuitable,))).search(request())

    assert result.status == ResultStatus.SUCCESS
    assert result.data is not None and result.data.no_suitable_validation_dataset
    assert result.data.suitable_datasets == ()
    assert "No suitable GEO" in result.warnings[0]
    assert not result.data.assessments[0].suitable


def test_missing_platform_fails_assay_platform_criterion() -> None:
    result = GEOClient(backend=FakeBackend((record("GSE200002", platform_accessions=()),))).search(request())
    criterion = next(c for c in result.data.assessments[0].criteria if c.criterion == "assay_platform")
    assert criterion.status == CriterionStatus.FAIL
    assert result.data.no_suitable_validation_dataset


def test_missing_semantic_metadata_fails_closed_without_llm() -> None:
    ambiguous = record("GSE300001", disease_terms=(), tissue_terms=())

    result = GEOClient(backend=FakeBackend((ambiguous,))).search(request())

    assert result.data is not None and result.data.no_suitable_validation_dataset
    statuses = {criterion.criterion: criterion.status for criterion in result.data.assessments[0].criteria}
    assert statuses["cancer_disease_match"] == CriterionStatus.AMBIGUOUS
    assert statuses["tissue_match"] == CriterionStatus.AMBIGUOUS


def test_llm_only_resolves_semantic_ambiguity_and_cannot_override_hard_failure() -> None:
    ambiguous = record(
        "GSE300002",
        disease_terms=(),
        tissue_terms=(),
        control_sample_count=2,
    )
    provider = MockLLMProvider(
        [{"decisions": [{"accession": "GSE300002", "disease_match": True, "tissue_match": True, "explanation": "Summary explicitly describes matched lung tissue."}]}]
    )

    result = GEOClient(backend=FakeBackend((ambiguous,)), semantic_provider=provider).search(request())

    assert result.data is not None and result.data.no_suitable_validation_dataset
    assessment = result.data.assessments[0]
    assert assessment.semantic_review_used
    assert assessment.score == 5
    assert len(provider.calls) == 1
    assert next(c for c in assessment.criteria if c.criterion == "sample_size").status == CriterionStatus.FAIL


def test_structured_matches_do_not_call_semantic_provider() -> None:
    provider = MockLLMProvider([RuntimeError("must not be called")])
    result = GEOClient(backend=FakeBackend((record("GSE400001"),)), semantic_provider=provider).search(request())
    assert result.status == ResultStatus.SUCCESS
    assert provider.calls == []


def test_backend_failure_is_structured() -> None:
    result = GEOClient(backend=FakeBackend(error=OSError("offline"))).search(request())
    assert result.status == ResultStatus.FAILURE
    assert result.error is not None and result.error.code == "geo_search_failed"
    assert result.provenance[0].source == "NCBI GEO DataSets"


def test_query_and_registry_use_typed_geo_request() -> None:
    backend = FakeBackend()
    registry = build_default_registry(geo_client=GEOClient(backend=backend))
    result = registry.execute(ToolName.SEARCH_GEO, {"cancer": "breast cancer", "tissue": "breast"})
    assert result.status == ResultStatus.SUCCESS
    assert "gse[Entry Type]" in build_geo_query(request())
    assert backend.calls[0][1] == 20
