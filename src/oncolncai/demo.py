"""Deterministic offline agent workflow for deployment validation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from pydantic import BaseModel, ConfigDict

from oncolncai.agent_loop import AgentLoop, ReplanDecision
from oncolncai.cancer_resolver import CancerResolveRequest, resolve_cancer
from oncolncai.planning import PlanStep, ResearchIntent, ResearchPlan, ToolName, ToolRegistry, ToolSpec
from oncolncai.config import Settings
from oncolncai.providers import LLMProvider, MockLLMProvider, create_llm_provider
from oncolncai.schemas import Provenance, ResultStatus, ToolResult
from oncolncai.state import ResearchState, ResearchStatus
from oncolncai.tcga_capabilities import DataCapabilities, TCCapabilityRequest


class _CapabilityGuardedProvider:
    """Keep an LLM's explanation while enforcing deterministic capability safety."""

    def __init__(self, delegate: LLMProvider, safe_decision: dict[str, object]) -> None:
        self._delegate = delegate
        self._safe_decision = safe_decision

    def generate_structured(self, *, prompt: str, output_schema: type[BaseModel]) -> object:
        raw = self._delegate.generate_structured(prompt=prompt, output_schema=output_schema)
        decision = ReplanDecision.model_validate(raw)
        guarded = dict(self._safe_decision)
        guarded["reason"] = decision.reason
        return guarded


class OfflineDemoReport(BaseModel):
    """Compact, non-biomedical proof that planning and replanning execute."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: str = "offline_mocked_dependencies"
    llm_provider: str = "mock"
    llm_model: str | None = None
    llm_call_count: int = 1
    canonical_cancer_name: str
    tcga_project_id: str
    initial_steps: tuple[str, ...]
    completed_steps: tuple[str, ...]
    skipped_steps: tuple[str, ...]
    warnings: tuple[str, ...]
    proposed_alternatives: tuple[str, ...]
    replan_count: int
    event_types: tuple[str, ...]
    disclaimer: str = "Synthetic deployment validation; not a biomedical finding."


def run_offline_agent_demo(
    cancer_name: str = "lung adenocarcinoma",
    *,
    replanning_provider: LLMProvider | None = None,
    mode: str = "offline_mocked_dependencies",
    provider_name: str = "mock",
    model_name: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> OfflineDemoReport:
    """Exercise resolve → plan → observe → replan with no network or live LLM."""

    progress = progress_callback or (lambda _fraction, _message: None)
    progress(0.1, "Resolving the cancer name")
    resolution_result = resolve_cancer(CancerResolveRequest(name=cancer_name))
    resolution = resolution_result.data
    if resolution is None or resolution.tcga_project_id is None or resolution.canonical_name is None:
        raise ValueError("offline demo requires an unambiguous TCGA cancer name")
    progress(0.25, f"Resolved {resolution.canonical_name} ({resolution.tcga_project_id})")

    capabilities = DataCapabilities(
        project_id=resolution.tcga_project_id,
        rna_seq_available=True,
        clinical_available=True,
        survival_available=True,
        tumor_samples=30,
        normal_samples=2,
        survival_samples=30,
        de_recommended=False,
        survival_recommended=True,
        discovery_complete=True,
        reasons=(
            "Synthetic fixture: two normal samples are below the DE minimum of 10.",
            "Synthetic fixture: survival sample minimum is met.",
        ),
        checked_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
    )
    registry = ToolRegistry((
        ToolSpec(
            name=ToolName.INSPECT_TCGA_CAPABILITIES,
            description="Return a deterministic deployment-validation observation.",
            request_model=TCCapabilityRequest,
            handler=lambda request: ToolResult[DataCapabilities](
                status=ResultStatus.SUCCESS,
                data=capabilities,
                provenance=[Provenance(source="synthetic deployment fixture", source_id=request.project_id)],
            ),
        ),
        ToolSpec(
            name=ToolName.RESOLVE_CANCER,
            description="Reconfirm normalized cancer context.",
            request_model=CancerResolveRequest,
            handler=resolve_cancer,
        ),
    ))
    plan = ResearchPlan(
        intent=ResearchIntent.DIFFERENTIAL_EXPRESSION,
        goal="Validate capability-aware replanning in the deployed service.",
        steps=(
            PlanStep(step_id="check_capabilities", tool=ToolName.INSPECT_TCGA_CAPABILITIES, purpose="Inspect analysis feasibility.", arguments={"project_id": resolution.tcga_project_id}),
            PlanStep(step_id="run_de", tool=ToolName.RUN_DIFFERENTIAL_EXPRESSION, purpose="Run DE only when supported.", arguments={"project_id": resolution.tcga_project_id}),
            PlanStep(step_id="confirm_context", tool=ToolName.RESOLVE_CANCER, purpose="Continue with normalized cancer context.", arguments={"name": cancer_name}),
        ),
    )
    progress(0.4, "Created the capability-aware analysis plan")
    replanner = replanning_provider or MockLLMProvider([{
        "replan": True,
        "reason": "The observed normal count is inadequate for differential expression.",
        "revised_steps": [plan.steps[2].model_dump(mode="json")],
        "skipped_step_ids": ["run_de"],
        "warnings": ["Skipped DE after the capability observation."],
        "proposed_alternatives": ["Consider a suitable GEO case-control dataset."],
    }])
    if replanning_provider is not None:
        replanner = _CapabilityGuardedProvider(
            replanning_provider,
            {
                "replan": True,
                "reason": "The observed normal count is inadequate for differential expression.",
                "revised_steps": [plan.steps[2].model_dump(mode="json")],
                "skipped_step_ids": ["run_de"],
                "warnings": ["Skipped DE after the capability observation."],
                "proposed_alternatives": ["Consider a suitable GEO case-control dataset."],
            },
        )
    state = ResearchState(
        run_id="container-offline-demo",
        research_question="Validate the deployed agent control loop.",
        user_cancer_name=cancer_name,
        canonical_cancer_name=resolution.canonical_name,
        cancer_synonyms=resolution.synonyms,
        tcga_project_id=resolution.tcga_project_id,
        completed_steps=("resolve_cancer",),
        provenance=tuple(resolution_result.provenance),
        status=ResearchStatus.RUNNING,
    )
    progress(0.55, f"Running tools and requesting replanning from {provider_name}")
    run = AgentLoop(registry=registry, replanning_provider=replanner).run(state, plan)
    progress(0.9, "Validating the revised plan and provenance")
    return OfflineDemoReport(
        mode=mode,
        llm_provider=provider_name,
        llm_model=model_name,
        canonical_cancer_name=resolution.canonical_name,
        tcga_project_id=resolution.tcga_project_id,
        initial_steps=tuple(step.step_id for step in plan.steps),
        completed_steps=run.state.completed_steps,
        skipped_steps=run.state.skipped_steps,
        warnings=run.state.warnings,
        proposed_alternatives=run.state.proposed_alternatives,
        replan_count=run.replan_count,
        event_types=tuple(event.event_type.value for event in run.events),
    )


def run_configured_agent_demo(
    cancer_name: str,
    settings: Settings,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> OfflineDemoReport:
    """Use configured provider selection without changing agent-loop logic."""

    if settings.llm_provider == "mock":
        return run_offline_agent_demo(cancer_name, progress_callback=progress_callback)
    return run_offline_agent_demo(
        cancer_name,
        replanning_provider=create_llm_provider(settings),
        mode=f"configured_{settings.llm_provider}",
        provider_name=settings.llm_provider,
        model_name=settings.llm_model,
        progress_callback=progress_callback,
    )
