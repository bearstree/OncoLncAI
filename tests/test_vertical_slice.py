import json
from pathlib import Path
from tempfile import TemporaryDirectory

from oncolncai import (
    CountFeature,
    DifferentialExpressionRequest,
    ExpressionGroup,
    ExpressionSample,
    GDCCapabilityClient,
    MockLLMProvider,
    PubMedClient,
    PubMedSearchRequest,
    ResearchState,
    ResultStatus,
    SurvivalRequest,
    SurvivalSample,
    TCCapabilityRequest,
    TCGASurvivalDataRequest,
    TraceEventType,
    ToolResult,
    VerticalSliceRequest,
    VerticalSliceRunner,
    load_trace,
)
from oncolncai.tcga_survival import CohortPreparationSummary, PreparedTCGASurvivalCohort
from datetime import datetime, timezone


ABSTRACT = "NEAT1 expression was associated with overall survival in lung adenocarcinoma."


class FakePubMedTransport:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, params: dict[str, str], timeout: float) -> bytes:
        self.calls += 1
        if url.endswith("esearch.fcgi"):
            return json.dumps(
                {"esearchresult": {"count": "1", "idlist": ["12345678"]}}
            ).encode()
        return f"""<?xml version="1.0"?>
        <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345678</PMID>
        <Article><ArticleTitle>NEAT1 in LUAD</ArticleTitle>
        <Abstract><AbstractText>{ABSTRACT}</AbstractText></Abstract>
        <Journal><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue>
        <Title>Fixture Journal</Title></Journal></Article>
        </MedlineCitation></PubmedArticle></PubmedArticleSet>""".encode()


class FakeGDCTransport:
    def __init__(self) -> None:
        self.calls = 0

    def post_json(self, url: str, payload: dict, timeout: float) -> dict:
        self.calls += 1
        if url.endswith("/files"):
            hits = [
                {
                    "file_id": f"tumor-{index}",
                    "cases": [{"samples": [{"submitter_id": f"T-{index}", "sample_type": "Primary Tumor"}]}],
                }
                for index in range(30)
            ] + [
                {
                    "file_id": f"normal-{index}",
                    "cases": [{"samples": [{"submitter_id": f"N-{index}", "sample_type": "Solid Tissue Normal"}]}],
                }
                for index in range(2)
            ]
        else:
            hits = [
                {
                    "case_id": f"case-{index}",
                    "demographic": {
                        "vital_status": "Dead" if index % 2 else "Alive",
                        "days_to_death": 200 + index if index % 2 else None,
                        "days_to_last_follow_up": 400 + index if not index % 2 else None,
                    },
                }
                for index in range(30)
            ]
        return {"data": {"hits": hits, "pagination": {"total": len(hits)}}}


def survival_request() -> SurvivalRequest:
    samples = tuple(
        SurvivalSample(
            sample_id=f"low-{index}", time=float(index + 2), event=int(index % 2 == 0), expression=float(index)
        )
        for index in range(20)
    ) + tuple(
        SurvivalSample(
            sample_id=f"high-{index}", time=float(index + 2), event=int(index % 2 == 0), expression=float(index + 20)
        )
        for index in range(20)
    )
    return SurvivalRequest(
        project_id="TCGA-LUAD",
        gene="NEAT1",
        samples=samples,
        cohort_source="fixture TCGA clinical",
        expression_source="fixture TCGA expression",
    )


def de_request() -> DifferentialExpressionRequest:
    samples = tuple(
        ExpressionSample(sample_id=f"case-{index}", group=ExpressionGroup.CASE)
        for index in range(20)
    ) + tuple(
        ExpressionSample(sample_id=f"control-{index}", group=ExpressionGroup.CONTROL)
        for index in range(10)
    )
    return DifferentialExpressionRequest(
        project_id="TCGA-LUAD",
        samples=samples,
        features=(
            CountFeature(feature_id="ENSG00000245532", gene_symbol="NEAT1", counts=(100,) * 20 + (20,) * 10),
            CountFeature(feature_id="BACKGROUND", gene_symbol="BG", counts=(50,) * 30),
        ),
        candidate_genes=("NEAT1",),
        count_source="fixture TCGA counts",
        annotation_source="fixture annotation",
    )


class FakeTCGASurvivalClient:
    def __init__(self, request: SurvivalRequest) -> None:
        self.request = request
        self.calls = 0

    def prepare(self, request: TCGASurvivalDataRequest):
        self.calls += 1
        cohort = PreparedTCGASurvivalCohort(
            project_id=request.project_id,
            gene=request.gene,
            ensembl_gene_id=request.ensembl_gene_id,
            endpoint=request.endpoint,
            expression_scale=request.expression_scale,
            samples=self.request.samples,
            preparation=CohortPreparationSummary(
                expression_columns=40, primary_tumor_columns=40, unique_primary_patients=40,
                clinical_project_rows=40, clinical_patients_with_endpoint=40, aligned_samples=40,
                aligned_events=20, excluded_non_primary=0, excluded_duplicate_aliquots=0,
                excluded_missing_endpoint=0, excluded_unmatched_expression=0,
            ),
            expression_url="https://example.test/expression.tsv.gz",
            clinical_url="https://example.test/clinical.tsv",
            expression_sha256="a" * 64,
            clinical_sha256="b" * 64,
            retrieved_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
        )
        return ToolResult[PreparedTCGASurvivalCohort](status=ResultStatus.SUCCESS, data=cohort)


def execute_vertical_slice_demo(trace_root: Path, *, automatic_survival: bool = False):
    survival = survival_request()
    initial_steps = [
        {
            "step_id": "search_literature",
            "tool": "search_pubmed",
            "purpose": "Retrieve source literature.",
            "arguments": PubMedSearchRequest(
                cancer="Lung Adenocarcinoma", concepts=("lncRNA",), max_results=3
            ).model_dump(mode="json"),
        },
        {
            "step_id": "check_capabilities",
            "tool": "inspect_tcga_capabilities",
            "purpose": "Determine valid TCGA analyses.",
            "arguments": TCCapabilityRequest(project_id="TCGA-LUAD").model_dump(mode="json"),
        },
        {
            "step_id": "run_de",
            "tool": "run_differential_expression",
            "purpose": "Test tumor versus normal expression.",
            "arguments": {"input_id": "luad_de"},
        },
        {
            "step_id": "run_survival",
            "tool": "run_survival_analysis",
            "purpose": "Test association with overall survival.",
            "arguments": {"input_id": "luad_survival"},
        },
    ]
    planning_provider = MockLLMProvider(
        [{
            "intent": "discover_biomarkers",
            "goal": "Find and validate a LUAD lncRNA candidate.",
            "steps": initial_steps,
            "assumptions": [],
            "requires_user_input": False,
        }]
    )
    replanning_provider = MockLLMProvider(
        [{
            "replan": True,
            "reason": "Two normal samples are inadequate for DE; survival remains valid.",
            "revised_steps": [initial_steps[3]],
            "skipped_step_ids": ["run_de"],
            "warnings": ["Skipped TCGA DE because only two normal samples were discovered."],
            "proposed_alternatives": ["Consider GEO validation in a later milestone."],
        }]
    )
    evidence_provider = MockLLMProvider(
        [{
            "records": [{
                "lncrna": "NEAT1",
                "cancer": "Lung Adenocarcinoma",
                "pmid": "12345678",
                "passage_id": "pmid:12345678:passage:1",
                "evidence_type": "prognostic",
                "biomarker_role": "prognostic",
                "outcome": "overall survival",
                "association_direction": "adverse",
                "sample_type": "tumor tissue",
                "supporting_text": ABSTRACT,
                "confidence": 0.95,
            }]
        }]
    )
    pubmed_transport = FakePubMedTransport()
    gdc_transport = FakeGDCTransport()
    survival_client = FakeTCGASurvivalClient(survival)
    runner = VerticalSliceRunner(
        planning_provider=planning_provider,
        evidence_provider=evidence_provider,
        replanning_provider=replanning_provider,
        pubmed_client=PubMedClient(transport=pubmed_transport, cache_dir=None),
        gdc_client=GDCCapabilityClient(transport=gdc_transport, cache_dir=None),
        tcga_survival_client=survival_client,
        trace_directory=trace_root / "traces",
    )

    result = runner.run(
        VerticalSliceRequest(
            run_id="vertical-001",
            research_question="Which lncRNAs are promising biomarkers for lung adenocarcinoma?",
            cancer_name="lung adenocarcinoma",
            survival_inputs={} if automatic_survival else {"luad_survival": survival},
            tcga_survival_inputs=(
                {"luad_survival": TCGASurvivalDataRequest(
                    project_id="TCGA-LUAD", gene="NEAT1", ensembl_gene_id="ENSG00000245532"
                )}
                if automatic_survival else {}
            ),
            differential_expression_inputs={"luad_de": de_request()},
        )
    )

    assert result.status == ResultStatus.SUCCESS, result.error
    assert result.data is not None
    run = result.data
    assert run.state.status.value == "completed"
    assert run.state.tcga_project_id == "TCGA-LUAD"
    assert run.state.candidate_lncrnas == ("NEAT1",)
    assert len(run.retrieved_papers) == 1
    assert run.retrieved_papers[0].pmid == "12345678"
    assert len(run.literature_evidence) == 1
    assert run.replan_count == 1
    assert run.state.skipped_steps == ("run_de",)
    assert len(run.de_results) == 0
    assert len(run.survival_results) == 1
    assert run.survival_results[0].sample_count == 40
    assert run.verification.passed is True
    assert run.final_result.verification_passed is True
    assert {claim.pmid for claim in run.final_result.claims} == {"12345678", None}
    assert "PMID: 12345678" in run.state.final_report
    assert "pubmed:12345678" in run.state.final_report
    assert "do not establish causation" in run.state.final_report
    assert "Consider GEO validation in a later milestone." in run.final_result.limitations
    assert "check_capabilities" in run.state.completed_steps
    assert "run_survival" in run.state.completed_steps
    assert "verify_claims" in run.state.completed_steps
    assert ResearchState.model_validate_json(run.state.model_dump_json()) == run.state
    assert pubmed_transport.calls == 2
    assert gdc_transport.calls == 2
    assert len(planning_provider.calls) == 1
    assert len(replanning_provider.calls) == 1
    assert len(evidence_provider.calls) == 1
    assert run.trace.status.value == "success"
    assert run.trace.tool_call_count == 5
    assert run.trace.model_call_count == 3
    assert run.trace.replan_count == 1
    assert any(event.event_type == TraceEventType.OBSERVATION for event in run.trace.events)
    assert any(
        event.event_type == TraceEventType.REPLAN and event.operation == "plan_revised"
        for event in run.trace.events
    )
    assert any(
        item.source_id == "TCGA-LUAD"
        for event in run.trace.events
        for item in event.provenance
    )
    assert run.trace_path is not None and run.trace_path.exists()
    assert load_trace(run.trace_path) == run.trace
    assert "low-0" not in planning_provider.calls[0][0]
    assert survival_client.calls == int(automatic_survival)
    return run


def test_end_to_end_replans_runs_survival_and_returns_verified_result() -> None:
    with TemporaryDirectory(dir=".") as directory:
        execute_vertical_slice_demo(Path(directory))


def test_end_to_end_can_prepare_supported_tcga_survival_input_automatically() -> None:
    with TemporaryDirectory(dir=".") as directory:
        run = execute_vertical_slice_demo(Path(directory), automatic_survival=True)
    assert run.survival_results[0].gene == "NEAT1"
    assert any("sha256:" in item.source for item in run.state.provenance)
