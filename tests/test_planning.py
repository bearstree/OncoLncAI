import pytest
from pydantic import ValidationError

from oncolncai import (
    MockLLMProvider,
    Planner,
    PlanningRequest,
    ResearchIntent,
    ResultStatus,
    ToolName,
    build_default_registry,
)


def plan_data(intent: str, steps: list[dict]) -> dict:
    return {
        "intent": intent,
        "goal": f"Complete the {intent} request with available tools.",
        "steps": steps,
        "assumptions": [],
        "requires_user_input": False,
    }


def step(step_id: str, tool: str, arguments: dict, purpose: str = "Gather evidence") -> dict:
    return {
        "step_id": step_id,
        "tool": tool,
        "purpose": purpose,
        "arguments": arguments,
    }


def test_different_intents_produce_different_valid_plans() -> None:
    literature_plan = plan_data(
        "literature_evidence",
        [step("search_literature", "search_pubmed", {"cancer": "BRCA", "concepts": ["lncRNA"]})],
    )
    prognosis_plan = plan_data(
        "prognostic_analysis",
        [step("check_survival", "inspect_tcga_capabilities", {"project_id": "TCGA-BRCA"})],
    )
    provider = MockLLMProvider([literature_plan, prognosis_plan])
    planner = Planner(provider=provider, registry=build_default_registry())

    literature = planner.plan(
        PlanningRequest(question="Find lncRNA literature for breast cancer")
    )
    prognosis = planner.plan(
        PlanningRequest(question="Can MALAT1 be tested for prognosis in BRCA?")
    )

    assert literature.status == ResultStatus.SUCCESS
    assert prognosis.status == ResultStatus.SUCCESS
    assert literature.data is not None and prognosis.data is not None
    assert literature.data.intent == ResearchIntent.LITERATURE_EVIDENCE
    assert prognosis.data.intent == ResearchIntent.PROGNOSTIC_ANALYSIS
    assert literature.data.steps != prognosis.data.steps
    assert len(provider.calls) == 2
    assert provider.calls[0][1].__name__ == "ResearchPlan"


@pytest.mark.parametrize(
    "intent",
    [
        "discover_biomarkers",
        "validate_lncrna",
        "differential_expression",
        "prognostic_analysis",
        "literature_evidence",
    ],
)
def test_all_required_intents_are_bounded(intent: str) -> None:
    request = PlanningRequest(question="test", intent_hint=intent)
    assert request.intent_hint is not None
    assert request.intent_hint.value == intent


def test_unknown_tool_in_model_output_is_rejected() -> None:
    provider = MockLLMProvider(
        [plan_data("literature_evidence", [step("browse", "arbitrary_shell", {})])]
    )
    planner = Planner(provider=provider, registry=build_default_registry())

    result = planner.plan(PlanningRequest(question="Find evidence"))

    assert result.status == ResultStatus.FAILURE
    assert result.error is not None and result.error.code == "invalid_plan"


def test_registered_tool_arguments_are_validated_during_planning() -> None:
    provider = MockLLMProvider(
        [
            plan_data(
                "prognostic_analysis",
                [step("check", "inspect_tcga_capabilities", {"project_id": "BRCA"})],
            )
        ]
    )
    planner = Planner(provider=provider, registry=build_default_registry())

    result = planner.plan(PlanningRequest(question="Check BRCA capabilities"))

    assert result.status == ResultStatus.FAILURE
    assert result.error is not None and "TCGA-" in result.error.message


def test_registry_executes_only_allow_listed_tools_with_typed_input() -> None:
    registry = build_default_registry()

    result = registry.execute(ToolName.RESOLVE_CANCER, {"name": "BRCA"})

    assert result.data is not None
    assert result.data.tcga_project_id == "TCGA-BRCA"
    with pytest.raises(ValueError, match="not allowed"):
        registry.execute("run_python", {})
    with pytest.raises(ValidationError):
        registry.execute(ToolName.RESOLVE_CANCER, {"unknown": "BRCA"})


def test_unconfigured_evidence_handler_cannot_execute() -> None:
    registry = build_default_registry()

    with pytest.raises(RuntimeError, match="no configured handler"):
        registry.execute(
            ToolName.EXTRACT_LITERATURE_EVIDENCE,
            {
                "source": {
                    "pmid": "123",
                    "cancer": "BRCA",
                    "title": "title",
                    "text": "passage",
                }
            },
        )


def test_plan_rejects_duplicate_step_ids() -> None:
    duplicate_steps = [
        step("resolve", "resolve_cancer", {"name": "BRCA"}),
        step("resolve", "search_pubmed", {"cancer": "BRCA"}),
    ]
    provider = MockLLMProvider([plan_data("discover_biomarkers", duplicate_steps)])
    planner = Planner(provider=provider, registry=build_default_registry())

    result = planner.plan(PlanningRequest(question="Discover BRCA biomarkers"))

    assert result.status == ResultStatus.FAILURE
    assert result.error is not None and "unique" in result.error.message


def test_provider_failure_is_structured() -> None:
    planner = Planner(
        provider=MockLLMProvider([RuntimeError("model unavailable")]),
        registry=build_default_registry(),
    )

    result = planner.plan(PlanningRequest(question="Find evidence"))

    assert result.status == ResultStatus.FAILURE
    assert result.error is not None
    assert result.error.code == "planner_provider_failed"


def test_planner_prompt_contains_exact_tool_argument_schema() -> None:
    provider = MockLLMProvider(
        [plan_data("literature_evidence", [step("resolve", "resolve_cancer", {"name": "BRCA"})])]
    )
    planner = Planner(provider=provider, registry=build_default_registry())

    result = planner.plan(PlanningRequest(question="Resolve BRCA"))

    assert result.status == ResultStatus.SUCCESS
    prompt = provider.calls[0][0]
    assert "input_schema" in prompt
    assert "exact property names" in prompt
    assert "'name'" in prompt


def test_cancer_project_mismatch_is_rejected_before_execution() -> None:
    invalid = plan_data(
        "prognostic_analysis",
        [step("check", "inspect_tcga_capabilities", {"project_id": "TCGA-BRCA"})],
    )
    planner = Planner(
        provider=MockLLMProvider([invalid]),
        registry=build_default_registry(),
        repair_attempts=0,
    )

    result = planner.plan(
        PlanningRequest(
            question="Assess LUAD prognosis",
            cancer_name="Lung Adenocarcinoma",
            tcga_project_id="TCGA-LUAD",
        )
    )

    assert result.status == ResultStatus.FAILURE
    assert result.error is not None and result.error.code == "invalid_plan"
    assert "does not match resolved project" in result.error.message


def test_invalid_local_plan_gets_one_bounded_repair_attempt() -> None:
    invalid = plan_data(
        "prognostic_analysis",
        [step("check", "inspect_tcga_capabilities", {"project_id": "TCGA-BRCA"})],
    )
    valid = plan_data(
        "prognostic_analysis",
        [step("check", "inspect_tcga_capabilities", {"project_id": "TCGA-LUAD"})],
    )
    provider = MockLLMProvider([invalid, valid])
    planner = Planner(provider=provider, registry=build_default_registry())

    result = planner.plan(
        PlanningRequest(
            question="Assess LUAD prognosis",
            cancer_name="Lung Adenocarcinoma",
            tcga_project_id="TCGA-LUAD",
        )
    )

    assert result.status == ResultStatus.SUCCESS
    assert len(provider.calls) == 2
    assert "previous proposal was rejected" in provider.calls[1][0]
    assert "bounded repair" in result.warnings[0]
