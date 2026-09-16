"""Minimal plan/act/observe/update/replan control loop."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from oncolncai.cancer_resolver import CancerResolution, ResolutionStatus
from oncolncai.differential_expression import DifferentialExpressionResult
from oncolncai.planning import PlanStep, ResearchPlan, ToolName, ToolRegistry
from oncolncai.providers import LLMProvider
from oncolncai.schemas import ResultStatus, ToolResult
from oncolncai.state import Decision, Observation, ResearchState, ResearchStatus
from oncolncai.survival import SurvivalAnalysisResult
from oncolncai.tcga_capabilities import DataCapabilities


class AgentEventType(StrEnum):
    ACTION_SELECTED = "action_selected"
    TOOL_EXECUTED = "tool_executed"
    OBSERVATION_RECORDED = "observation_recorded"
    REPLAN_REQUESTED = "replan_requested"
    REPLAN_REPAIRED = "replan_repaired"
    PLAN_REVISED = "plan_revised"
    ACTION_FAILED = "action_failed"
    STOPPED = "stopped"


class AgentEvent(BaseModel):
    """External execution trace event, kept separate from research state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    event_type: AgentEventType
    step_id: str | None = None
    tool: ToolName | None = None
    summary: str = Field(min_length=1, max_length=500)


class ReplanDecision(BaseModel):
    """Bounded semantic decision returned after a capability limitation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    replan: bool
    reason: str = Field(min_length=1, max_length=500)
    revised_steps: tuple[PlanStep, ...] = Field(default=(), max_length=8)
    skipped_step_ids: tuple[str, ...] = Field(default=(), max_length=8)
    warnings: tuple[str, ...] = Field(default=(), max_length=8)
    proposed_alternatives: tuple[str, ...] = Field(default=(), max_length=5)

    @model_validator(mode="after")
    def validate_decision(self) -> ReplanDecision:
        if self.replan and not (self.revised_steps or self.skipped_step_ids):
            raise ValueError("replanning must revise or skip at least one step")
        if not self.replan and (self.revised_steps or self.skipped_step_ids):
            raise ValueError("a no-replan decision cannot revise or skip steps")
        revised_ids = [step.step_id for step in self.revised_steps]
        if len(revised_ids) != len(set(revised_ids)):
            raise ValueError("revised plan step IDs must be unique")
        if set(revised_ids) & set(self.skipped_step_ids):
            raise ValueError("a step cannot be both revised and skipped")
        return self


class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: ResearchState
    plan: ResearchPlan
    events: tuple[AgentEvent, ...]
    actions_executed: int = Field(ge=0)
    replan_count: int = Field(ge=0)
    finished: bool


class AgentLoop:
    """Execute registered actions and replan only after relevant observations."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        replanning_provider: LLMProvider,
        max_actions: int = 20,
    ) -> None:
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        self._registry = registry
        self._provider = replanning_provider
        self._max_actions = max_actions

    def run(self, state: ResearchState, plan: ResearchPlan) -> AgentRunResult:
        remaining = list(plan.steps)
        current_plan = plan
        events: list[AgentEvent] = []
        actions_executed = 0
        replan_count = 0
        state = state.updated(
            intent=plan.intent.value,
            initial_plan=state.initial_plan or tuple(step.step_id for step in remaining),
            research_plan=tuple(step.step_id for step in remaining),
            next_action=remaining[0].step_id if remaining else None,
            status=ResearchStatus.RUNNING,
        )

        while remaining and actions_executed < self._max_actions:
            step = remaining.pop(0)
            if step.step_id in state.completed_steps or step.step_id in state.skipped_steps:
                continue
            events.append(
                self._event(
                    events,
                    AgentEventType.ACTION_SELECTED,
                    f"Selected {step.tool.value}.",
                    step,
                )
            )
            if not self._registry.is_executable(step.tool):
                state = state.updated(
                    current_step=None,
                    next_action=step.step_id,
                    research_plan=tuple(item.step_id for item in (step, *remaining)),
                    status=ResearchStatus.PARTIAL,
                )
                events.append(
                    self._event(
                        events,
                        AgentEventType.STOPPED,
                        f"Retained {step.step_id}; its tool is not executable yet.",
                        step,
                    )
                )
                break

            state = state.updated(current_step=step.step_id, next_action=step.step_id)
            try:
                result = self._registry.execute(step.tool, step.arguments)
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                state = state.updated(
                    failed_steps=(*state.failed_steps, step.step_id),
                    warnings=(*state.warnings, message),
                    current_step=None,
                    next_action=remaining[0].step_id if remaining else None,
                    status=ResearchStatus.PARTIAL,
                )
                events.append(
                    self._event(events, AgentEventType.ACTION_FAILED, message, step)
                )
                continue

            actions_executed += 1
            events.append(
                self._event(events, AgentEventType.TOOL_EXECUTED, f"Executed {step.tool.value}.", step)
            )
            if not isinstance(result, ToolResult) or result.status == ResultStatus.FAILURE:
                message = (
                    result.error.message
                    if isinstance(result, ToolResult) and result.error is not None
                    else "Tool returned an invalid or failed result."
                )
                state = state.updated(
                    failed_steps=(*state.failed_steps, step.step_id),
                    warnings=(*state.warnings, message),
                    current_step=None,
                    next_action=remaining[0].step_id if remaining else None,
                    status=ResearchStatus.PARTIAL,
                )
                events.append(
                    self._event(events, AgentEventType.ACTION_FAILED, message, step)
                )
                continue

            state = self._record_success(state, step, result)
            events.append(
                self._event(
                    events,
                    AgentEventType.OBSERVATION_RECORDED,
                    state.observations[-1].summary,
                    step,
                )
            )

            capabilities = result.data if isinstance(result.data, DataCapabilities) else None
            if capabilities is not None and self._capabilities_require_replan(
                capabilities, remaining
            ):
                events.append(
                    self._event(
                        events,
                        AgentEventType.REPLAN_REQUESTED,
                        "Capability observation invalidated at least one planned analysis.",
                        step,
                    )
                )
                decision_result = self._request_replan(state, current_plan, remaining, capabilities)
                if isinstance(decision_result, str):
                    state = state.updated(
                        warnings=(*state.warnings, decision_result),
                        current_step=None,
                        status=ResearchStatus.FAILED,
                    )
                    events.append(
                        self._event(events, AgentEventType.STOPPED, decision_result, step)
                    )
                    break
                decision = decision_result
                if decision.replan:
                    try:
                        self._validate_revision(
                            state, remaining, decision, capabilities
                        )
                    except ValueError as exc:
                        repaired = self._request_replan(
                            state,
                            current_plan,
                            remaining,
                            capabilities,
                            repair_reason=str(exc),
                        )
                        if isinstance(repaired, str):
                            state = state.updated(
                                warnings=(*state.warnings, str(exc), repaired),
                                current_step=None,
                                status=ResearchStatus.FAILED,
                            )
                            events.append(self._event(events, AgentEventType.STOPPED, repaired, step))
                            break
                        try:
                            self._validate_revision(state, remaining, repaired, capabilities)
                        except ValueError as repair_exc:
                            state = state.updated(
                                warnings=(*state.warnings, str(exc), str(repair_exc)),
                                current_step=None,
                                status=ResearchStatus.FAILED,
                            )
                            events.append(self._event(events, AgentEventType.STOPPED, str(repair_exc), step))
                            break
                        decision = repaired
                        events.append(
                            self._event(
                                events,
                                AgentEventType.REPLAN_REPAIRED,
                                "Accepted one validated replanning repair after rejecting the first proposal.",
                                step,
                            )
                        )
                    remaining = list(decision.revised_steps)
                    current_plan = ResearchPlan.model_validate(
                        {
                            **current_plan.model_dump(),
                            "steps": tuple(decision.revised_steps),
                        }
                    )
                    state = state.updated(
                        skipped_steps=(*state.skipped_steps, *decision.skipped_step_ids),
                        warnings=(*state.warnings, *decision.warnings),
                        proposed_alternatives=(
                            *state.proposed_alternatives,
                            *decision.proposed_alternatives,
                        ),
                        decisions=(
                            *state.decisions,
                            Decision(decision="Plan revised", basis=decision.reason),
                        ),
                        replanning_history=(
                            *state.replanning_history,
                            Decision(decision="Plan revised", basis=decision.reason),
                        ),
                        research_plan=tuple(item.step_id for item in remaining),
                        next_action=remaining[0].step_id if remaining else None,
                    )
                    replan_count += 1
                    events.append(
                        self._event(
                            events,
                            AgentEventType.PLAN_REVISED,
                            decision.reason,
                            step,
                        )
                    )

        else:
            if remaining:
                warning = f"Stopped after reaching the {self._max_actions}-action limit."
                state = state.updated(
                    warnings=(*state.warnings, warning),
                    current_step=None,
                    next_action=remaining[0].step_id,
                    status=ResearchStatus.PARTIAL,
                )
                events.append(self._event(events, AgentEventType.STOPPED, warning))
            else:
                state = state.updated(
                    current_step=None,
                    next_action=None,
                    research_plan=(),
                    status=ResearchStatus.PARTIAL,
                )

        finished = state.next_action is None and state.status != ResearchStatus.FAILED
        return AgentRunResult(
            state=state,
            plan=current_plan,
            events=tuple(events),
            actions_executed=actions_executed,
            replan_count=replan_count,
            finished=finished,
        )

    @staticmethod
    def _event(
        events: list[AgentEvent],
        event_type: AgentEventType,
        summary: str,
        step: PlanStep | None = None,
    ) -> AgentEvent:
        return AgentEvent(
            sequence=len(events) + 1,
            event_type=event_type,
            step_id=step.step_id if step else None,
            tool=step.tool if step else None,
            summary=summary,
        )

    @staticmethod
    def _record_success(
        state: ResearchState, step: PlanStep, result: ToolResult
    ) -> ResearchState:
        data = result.data
        changes: dict[str, object] = {}
        if isinstance(data, DataCapabilities):
            summary = (
                f"Observed {data.tumor_samples} tumor, {data.normal_samples} normal, "
                f"and {data.survival_samples} survival-usable samples; "
                f"DE recommended={data.de_recommended}, "
                f"survival recommended={data.survival_recommended}."
            )
            changes["data_capabilities"] = data
        elif isinstance(data, CancerResolution):
            summary = f"Cancer resolution status: {data.status.value}."
            if data.status == ResolutionStatus.RESOLVED:
                changes.update(
                    user_cancer_name=data.query,
                    canonical_cancer_name=data.canonical_name,
                    cancer_synonyms=data.synonyms,
                    tcga_project_id=data.tcga_project_id,
                )
        elif isinstance(data, SurvivalAnalysisResult):
            summary = (
                f"Survival analysis completed for {data.gene}: n={data.sample_count}, "
                f"events={data.event_count}, HR={data.hazard_ratio:.4g}, "
                f"Cox p={data.cox_p_value:.4g}."
            )
            changes["survival_results"] = (*state.survival_results, data)
        elif isinstance(data, DifferentialExpressionResult):
            summary = (
                f"Differential expression completed with {data.case_sample_count} cases, "
                f"{data.control_sample_count} controls, and {data.tested_feature_count} "
                "tested features."
            )
            changes["de_results"] = (*state.de_results, data)
        else:
            summary = f"{step.tool.value} completed successfully."
        return state.updated(
            **changes,
            completed_steps=(*state.completed_steps, step.step_id),
            observations=(
                *state.observations,
                Observation(
                    action=step.tool.value,
                    summary=summary,
                    provenance=tuple(result.provenance),
                ),
            ),
            provenance=(*state.provenance, *result.provenance),
            warnings=(*state.warnings, *result.warnings),
            current_step=None,
        )

    @staticmethod
    def _capabilities_require_replan(
        capabilities: DataCapabilities, remaining: list[PlanStep]
    ) -> bool:
        return any(
            (step.tool == ToolName.RUN_DIFFERENTIAL_EXPRESSION and not capabilities.de_recommended)
            or (step.tool == ToolName.RUN_SURVIVAL_ANALYSIS and not capabilities.survival_recommended)
            for step in remaining
        )

    def _request_replan(
        self,
        state: ResearchState,
        plan: ResearchPlan,
        remaining: list[PlanStep],
        capabilities: DataCapabilities,
        repair_reason: str | None = None,
    ) -> ReplanDecision | str:
        prompt = (
            "Revise only the remaining plan in response to this capability observation. "
            "Preserve valid analyses, explicitly skip invalid analyses, and propose later alternatives "
            "without pretending unavailable tools can execute.\n"
            f"Goal: {plan.goal}\n"
            f"Completed steps: {state.completed_steps}\n"
            f"Remaining steps: {[step.model_dump(mode='json') for step in remaining]}\n"
            f"Capabilities: {capabilities.model_dump_json()}"
        )
        if repair_reason is not None:
            prompt += (
                "\nThe previous revision was rejected before execution. Regenerate one corrected "
                f"revision. Validation error: {repair_reason}"
            )
        try:
            raw = self._provider.generate_structured(
                prompt=prompt,
                output_schema=ReplanDecision,
            )
            return ReplanDecision.model_validate(raw)
        except (ValidationError, ValueError) as exc:
            return f"Invalid replanning decision: {exc}"
        except Exception as exc:
            return f"Replanning provider failed: {str(exc) or type(exc).__name__}"

    @staticmethod
    def _validate_revision(
        state: ResearchState,
        previous_remaining: list[PlanStep],
        decision: ReplanDecision,
        capabilities: DataCapabilities,
    ) -> None:
        previous_ids = {step.step_id for step in previous_remaining}
        revised_ids = {step.step_id for step in decision.revised_steps}
        skipped_ids = set(decision.skipped_step_ids)
        if not skipped_ids <= previous_ids:
            raise ValueError("replanning may skip only remaining steps")
        if revised_ids & set(state.completed_steps):
            raise ValueError("replanning cannot repeat completed steps")
        if not revised_ids <= previous_ids:
            raise ValueError("replanning cannot add executable steps in this milestone")
        if previous_ids != revised_ids | skipped_ids:
            raise ValueError("replanning must retain or explicitly skip every remaining step")
        previous_tools = {step.step_id: step.tool for step in previous_remaining}
        if any(previous_tools[step.step_id] != step.tool for step in decision.revised_steps):
            raise ValueError("replanning cannot change the tool of a retained step")
        de_ids = {
            step.step_id
            for step in previous_remaining
            if step.tool == ToolName.RUN_DIFFERENTIAL_EXPRESSION
        }
        survival_ids = {
            step.step_id
            for step in previous_remaining
            if step.tool == ToolName.RUN_SURVIVAL_ANALYSIS
        }
        if not capabilities.de_recommended and not de_ids <= skipped_ids:
            raise ValueError("unsupported differential-expression steps must be skipped")
        if capabilities.survival_recommended and not survival_ids <= revised_ids:
            raise ValueError("supported survival steps must be retained")
