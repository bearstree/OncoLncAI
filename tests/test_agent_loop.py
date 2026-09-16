from datetime import datetime, timezone

from oncolncai import (
    AgentEventType,
    AgentLoop,
    CancerResolveRequest,
    DataCapabilities,
    MockLLMProvider,
    PlanStep,
    Provenance,
    ResearchIntent,
    ResearchPlan,
    ResearchState,
    ResearchStatus,
    ResultStatus,
    TCCapabilityRequest,
    ToolName,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    build_default_registry,
    resolve_cancer,
)


def capability_result(*, normals: int = 2) -> ToolResult[DataCapabilities]:
    capabilities = DataCapabilities(
        project_id="TCGA-LUAD",
        rna_seq_available=True,
        clinical_available=True,
        survival_available=True,
        tumor_samples=30,
        normal_samples=normals,
        survival_samples=30,
        de_recommended=normals >= 10,
        survival_recommended=True,
        discovery_complete=True,
        reasons=(
            f"DE not recommended: {normals} solid-tissue normals; minimum is 10."
            if normals < 10
            else "DE recommended: minimum tumor and normal sample counts are met.",
            "Survival recommended: minimum usable case count is met.",
        ),
        checked_at=datetime(2026, 9, 14, tzinfo=timezone.utc),
    )
    return ToolResult[DataCapabilities](
        status=ResultStatus.SUCCESS,
        data=capabilities,
        provenance=[Provenance(source="mock GDC", source_id="TCGA-LUAD")],
    )


def capability_registry() -> ToolRegistry:
    return ToolRegistry(
        (
            ToolSpec(
                name=ToolName.INSPECT_TCGA_CAPABILITIES,
                description="Return a synthetic capability observation.",
                request_model=TCCapabilityRequest,
                handler=lambda request: capability_result(),
            ),
        )
    )


def adaptive_plan() -> ResearchPlan:
    return ResearchPlan(
        intent=ResearchIntent.DISCOVER_BIOMARKERS,
        goal="Assess LUAD candidates with literature, DE, and survival evidence.",
        steps=(
            PlanStep(
                step_id="check_capabilities",
                tool=ToolName.INSPECT_TCGA_CAPABILITIES,
                purpose="Check valid TCGA analyses.",
                arguments={"project_id": "TCGA-LUAD"},
            ),
            PlanStep(
                step_id="run_de",
                tool=ToolName.RUN_DIFFERENTIAL_EXPRESSION,
                purpose="Compare primary tumors with normal tissue.",
                arguments={"project_id": "TCGA-LUAD"},
            ),
            PlanStep(
                step_id="run_survival",
                tool=ToolName.RUN_SURVIVAL_ANALYSIS,
                purpose="Assess prognostic relevance.",
                arguments={"project_id": "TCGA-LUAD"},
            ),
        ),
    )


def test_insufficient_normals_replans_and_preserves_successful_work() -> None:
    revised_survival = adaptive_plan().steps[2].model_dump(mode="json")
    provider = MockLLMProvider(
        [
            {
                "replan": True,
                "reason": "Two normal samples are insufficient for TCGA DE; survival remains supported.",
                "revised_steps": [revised_survival],
                "skipped_step_ids": ["run_de"],
                "warnings": ["Skipped TCGA DE because only two normal samples were found."],
                "proposed_alternatives": ["Consider GEO validation in a later milestone."],
            }
        ]
    )
    initial = ResearchState(
        run_id="adaptive-001",
        research_question="Which lncRNAs are promising in LUAD?",
        canonical_cancer_name="Lung Adenocarcinoma",
        tcga_project_id="TCGA-LUAD",
        completed_steps=("resolve_cancer", "retrieve_literature"),
        status=ResearchStatus.RUNNING,
    )

    run = AgentLoop(
        registry=capability_registry(), replanning_provider=provider
    ).run(initial, adaptive_plan())

    assert run.actions_executed == 1
    assert run.replan_count == 1
    assert run.finished is False
    assert run.state.completed_steps == (
        "resolve_cancer",
        "retrieve_literature",
        "check_capabilities",
    )
    assert run.state.skipped_steps == ("run_de",)
    assert run.state.research_plan == ("run_survival",)
    assert run.state.next_action == "run_survival"
    assert run.state.data_capabilities is not None
    assert run.state.data_capabilities.normal_samples == 2
    assert run.state.data_capabilities.survival_recommended is True
    assert "Consider GEO validation in a later milestone." in run.state.proposed_alternatives
    assert any("Skipped TCGA DE" in warning for warning in run.state.warnings)
    assert run.state.decisions[-1].decision == "Plan revised"
    assert [event.event_type for event in run.events] == [
        AgentEventType.ACTION_SELECTED,
        AgentEventType.TOOL_EXECUTED,
        AgentEventType.OBSERVATION_RECORDED,
        AgentEventType.REPLAN_REQUESTED,
        AgentEventType.PLAN_REVISED,
        AgentEventType.ACTION_SELECTED,
        AgentEventType.STOPPED,
    ]
    assert len(provider.calls) == 1
    assert "normal_samples\":2" in provider.calls[0][0]


def test_different_plan_executes_without_capability_replanning() -> None:
    plan = ResearchPlan(
        intent=ResearchIntent.LITERATURE_EVIDENCE,
        goal="Resolve the cancer name only.",
        steps=(
            PlanStep(
                step_id="resolve",
                tool=ToolName.RESOLVE_CANCER,
                purpose="Resolve BRCA.",
                arguments={"name": "BRCA"},
            ),
        ),
    )
    provider = MockLLMProvider([])
    state = ResearchState(run_id="simple-001", research_question="Resolve BRCA")

    run = AgentLoop(
        registry=build_default_registry(), replanning_provider=provider
    ).run(state, plan)

    assert run.finished is True
    assert run.replan_count == 0
    assert run.actions_executed == 1
    assert run.state.tcga_project_id == "TCGA-BRCA"
    assert run.state.completed_steps == ("resolve",)
    assert provider.calls == []


def test_invalid_revision_cannot_drop_a_step_silently() -> None:
    provider = MockLLMProvider(
        [
            {
                "replan": True,
                "reason": "Invalidly drop both analyses.",
                "revised_steps": [],
                "skipped_step_ids": ["run_de"],
                "warnings": [],
                "proposed_alternatives": [],
            }
        ]
    )
    state = ResearchState(run_id="invalid-001", research_question="Test guardrail")

    run = AgentLoop(
        registry=capability_registry(), replanning_provider=provider
    ).run(state, adaptive_plan())

    assert run.state.status == ResearchStatus.FAILED
    assert any("retain or explicitly skip every remaining step" in item for item in run.state.warnings)
    assert "check_capabilities" in run.state.completed_steps


def test_replanner_cannot_override_capability_guardrails() -> None:
    retained_de = adaptive_plan().steps[1].model_dump(mode="json")
    provider = MockLLMProvider(
        [
            {
                "replan": True,
                "reason": "Attempt to keep unsupported DE and remove survival.",
                "revised_steps": [retained_de],
                "skipped_step_ids": ["run_survival"],
                "warnings": [],
                "proposed_alternatives": [],
            }
        ]
    )

    run = AgentLoop(
        registry=capability_registry(), replanning_provider=provider
    ).run(
        ResearchState(run_id="guardrail-001", research_question="Test capabilities"),
        adaptive_plan(),
    )

    assert run.state.status == ResearchStatus.FAILED
    assert any(
        "unsupported differential-expression steps must be skipped" in warning
        for warning in run.state.warnings
    )


def test_independent_successful_work_survives_a_tool_failure() -> None:
    registry = ToolRegistry(
        (
            ToolSpec(
                name=ToolName.INSPECT_TCGA_CAPABILITIES,
                description="Fail in a structured way.",
                request_model=TCCapabilityRequest,
                handler=lambda request: ToolResult[DataCapabilities](
                    status=ResultStatus.FAILURE,
                    error={"code": "api_unavailable", "message": "GDC unavailable", "retryable": True},
                ),
            ),
            ToolSpec(
                name=ToolName.RESOLVE_CANCER,
                description="Resolve cancer independently.",
                request_model=CancerResolveRequest,
                handler=resolve_cancer,
            ),
        )
    )
    plan = ResearchPlan(
        intent=ResearchIntent.DISCOVER_BIOMARKERS,
        goal="Demonstrate partial completion.",
        steps=(
            PlanStep(step_id="gdc", tool=ToolName.INSPECT_TCGA_CAPABILITIES, purpose="Check GDC.", arguments={"project_id": "TCGA-LUAD"}),
            PlanStep(step_id="resolve", tool=ToolName.RESOLVE_CANCER, purpose="Resolve BRCA.", arguments={"name": "BRCA"}),
        ),
    )

    run = AgentLoop(registry=registry, replanning_provider=MockLLMProvider([])).run(
        ResearchState(run_id="partial-001", research_question="Test recovery"), plan
    )

    assert run.state.status == ResearchStatus.PARTIAL
    assert run.state.failed_steps == ("gdc",)
    assert run.state.completed_steps == ("resolve",)
    assert run.state.tcga_project_id == "TCGA-BRCA"
    assert run.finished is True
    assert AgentEventType.ACTION_FAILED in [event.event_type for event in run.events]


def test_invalid_replan_gets_one_validated_repair_attempt() -> None:
    invalid = {
        "replan": True,
        "reason": "Keep an unsupported analysis.",
        "revised_steps": [adaptive_plan().steps[1].model_dump(mode="json")],
        "skipped_step_ids": ["run_survival"],
        "warnings": [],
        "proposed_alternatives": [],
    }
    valid = {
        "replan": True,
        "reason": "Skip DE and retain survival.",
        "revised_steps": [adaptive_plan().steps[2].model_dump(mode="json")],
        "skipped_step_ids": ["run_de"],
        "warnings": ["DE skipped because normals are inadequate."],
        "proposed_alternatives": [],
    }
    provider = MockLLMProvider([invalid, valid])

    run = AgentLoop(registry=capability_registry(), replanning_provider=provider).run(
        ResearchState(run_id="repair-001", research_question="Test repair"), adaptive_plan()
    )

    assert run.state.status == ResearchStatus.PARTIAL
    assert run.state.skipped_steps == ("run_de",)
    assert len(provider.calls) == 2
    assert "previous revision was rejected" in provider.calls[1][0]
    assert AgentEventType.REPLAN_REPAIRED in [event.event_type for event in run.events]
