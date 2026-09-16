"""End-to-end orchestration of the implemented evidence-to-validation slice."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oncolncai.agent_loop import AgentEvent, AgentEventType, AgentLoop
from oncolncai.cancer_resolver import CancerResolveRequest, ResolutionStatus, resolve_cancer
from oncolncai.differential_expression import (
    DifferentialExpressionRequest,
    DifferentialExpressionResult,
    run_differential_expression,
)
from oncolncai.evidence import (
    EvidenceExtractionRequest,
    EvidenceRecord,
    LiteratureSource,
    StructuredGenerationProvider,
    extract_literature_evidence,
)
from oncolncai.planning import (
    Planner,
    PlanStep,
    PlanningRequest,
    ResearchPlan,
    ToolName,
    ToolRegistry,
    ToolSpec,
)
from oncolncai.providers import LLMProvider
from oncolncai.pubmed import (
    PubMedClient,
    PubMedRecord,
    PubMedSearchRequest,
    PubMedSearchResult,
    search_pubmed,
)
from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult
from oncolncai.state import Observation, ResearchState, ResearchStatus
from oncolncai.survival import SurvivalAnalysisResult, SurvivalRequest, run_survival_analysis
from oncolncai.tcga_capabilities import (
    GDCCapabilityClient,
    TCCapabilityRequest,
    inspect_tcga_capabilities,
)
from oncolncai.tcga_survival import TCGASurvivalDataClient, TCGASurvivalDataRequest
from oncolncai.tracing import (
    ExecutionTrace,
    TraceEventType,
    TraceRecorder,
    TraceStatus,
    save_trace,
)
from oncolncai.verifier import (
    ClaimKind,
    EvidenceReference,
    NumericAssertion,
    VerificationClaim,
    VerificationReport,
    VerificationRequest,
    verify_claims,
)


class VerticalSliceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    research_question: str = Field(min_length=1, max_length=2000)
    cancer_name: str = Field(min_length=1, max_length=200)
    survival_inputs: dict[str, SurvivalRequest] = Field(default_factory=dict)
    tcga_survival_inputs: dict[str, TCGASurvivalDataRequest] = Field(default_factory=dict)
    differential_expression_inputs: dict[str, DifferentialExpressionRequest] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_analysis_inputs(self) -> VerticalSliceRequest:
        identifiers = (
            *self.survival_inputs,
            *self.tcga_survival_inputs,
            *self.differential_expression_inputs,
        )
        if not identifiers:
            raise ValueError("at least one deterministic analysis input is required")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("analysis input IDs must be unique across analysis types")
        return self


class AnalysisInputReference(BaseModel):
    """Compact plan argument resolved to typed data outside the LLM context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class GroundedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    text: str
    source_ids: tuple[str, ...]
    pmid: str | None = None
    numeric_values: dict[str, float] = Field(default_factory=dict)


class GroundedFinalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    research_question: str
    canonical_cancer_name: str
    tcga_project_id: str
    candidate_lncrnas: tuple[str, ...]
    claims: tuple[GroundedClaim, ...]
    limitations: tuple[str, ...]
    verification_passed: bool
    disclaimer: str = "Research evidence only; not clinical decision support."


class VerticalSliceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: ResearchState
    initial_plan: ResearchPlan
    executed_plan: ResearchPlan
    events: tuple[AgentEvent, ...]
    retrieved_papers: tuple[PubMedRecord, ...]
    literature_evidence: tuple[EvidenceRecord, ...]
    survival_results: tuple[SurvivalAnalysisResult, ...]
    de_results: tuple[DifferentialExpressionResult, ...]
    verification: VerificationReport
    final_result: GroundedFinalResult
    replan_count: int = Field(ge=0)
    trace: ExecutionTrace
    trace_path: Path | None = None


class VerticalSliceRunner:
    """Coordinate existing tools while keeping their computation boundaries intact."""

    def __init__(
        self,
        *,
        planning_provider: LLMProvider,
        evidence_provider: StructuredGenerationProvider,
        replanning_provider: LLMProvider,
        semantic_verifier_provider: LLMProvider | None = None,
        pubmed_client: PubMedClient | None = None,
        gdc_client: GDCCapabilityClient | None = None,
        tcga_survival_client: TCGASurvivalDataClient | None = None,
        trace_directory: Path | None = None,
        trace_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._planning_provider = planning_provider
        self._evidence_provider = evidence_provider
        self._replanning_provider = replanning_provider
        self._semantic_verifier_provider = semantic_verifier_provider
        self._pubmed_client = pubmed_client
        self._gdc_client = gdc_client
        self._tcga_survival_client = tcga_survival_client or TCGASurvivalDataClient()
        self._trace_directory = trace_directory
        self._trace_clock = trace_clock

    def run(self, request: VerticalSliceRequest) -> ToolResult[VerticalSliceResult]:
        trace_recorder = TraceRecorder(request.run_id, clock=self._trace_clock)
        state = ResearchState(
            run_id=request.run_id,
            research_question=request.research_question,
            user_cancer_name=request.cancer_name,
        )
        registry = self._build_registry(request)
        resolution_result = resolve_cancer(CancerResolveRequest(name=request.cancer_name))
        resolution = resolution_result.data
        if resolution is None or resolution.status != ResolutionStatus.RESOLVED:
            return self._failure(
                "cancer_resolution_required",
                "The cancer name must resolve unambiguously before planning.",
                resolution_result.provenance,
            )
        state = state.updated(
            canonical_cancer_name=resolution.canonical_name,
            cancer_synonyms=resolution.synonyms,
            tcga_project_id=resolution.tcga_project_id,
            completed_steps=("resolve_cancer",),
            observations=(
                Observation(
                    action="resolve_cancer",
                    summary=f"Resolved to {resolution.canonical_name} ({resolution.tcga_project_id}).",
                    provenance=tuple(resolution_result.provenance),
                ),
            ),
            provenance=tuple(resolution_result.provenance),
            status=ResearchStatus.RUNNING,
        )
        trace_recorder.record(
            TraceEventType.TOOL_CALL,
            component="cancer_resolver",
            operation="resolve_cancer",
            status=TraceStatus.SUCCESS,
            summary=f"Resolved cancer to {resolution.tcga_project_id}.",
            step_id="resolve_cancer",
            tool="resolve_cancer",
            provenance=tuple(resolution_result.provenance),
        )

        plan_result = Planner(
            provider=self._planning_provider, registry=registry
        ).plan(
            PlanningRequest(
                question=request.research_question,
                cancer_name=resolution.canonical_name,
                tcga_project_id=resolution.tcga_project_id,
            )
        )
        if plan_result.status == ResultStatus.FAILURE or plan_result.data is None:
            return self._failure(
                "planning_failed",
                plan_result.error.message if plan_result.error else "Planning failed.",
                state.provenance,
            )
        plan = plan_result.data
        for warning in plan_result.warnings:
            trace_recorder.record(
                TraceEventType.MODEL_CALL,
                component="planner",
                operation="repair_plan",
                status=TraceStatus.SUCCESS,
                summary=warning,
            )
        trace_recorder.record(
            TraceEventType.MODEL_CALL,
            component="planner",
            operation="create_plan",
            status=TraceStatus.SUCCESS,
            summary=f"Created a validated {len(plan.steps)}-step plan.",
        )
        try:
            search_step, analysis_plan = self._validate_and_split_plan(
                registry, request, plan, resolution.canonical_name, resolution.tcga_project_id
            )
        except ValueError as exc:
            return self._failure("invalid_vertical_slice_plan", str(exc), state.provenance)
        state = state.updated(
            intent=plan.intent.value,
            research_plan=tuple(step.step_id for step in plan.steps),
            next_action=search_step.step_id,
        )

        search_result = registry.execute(search_step.tool, search_step.arguments)
        if not isinstance(search_result, ToolResult) or search_result.status == ResultStatus.FAILURE:
            return self._failure(
                "pubmed_retrieval_failed",
                self._result_error(search_result),
                (*state.provenance, *getattr(search_result, "provenance", ())),
            )
        if not isinstance(search_result.data, PubMedSearchResult):
            return self._failure("invalid_pubmed_result", "PubMed returned an unexpected payload.", state.provenance)
        papers = search_result.data.records
        state = self._record_stage(
            state,
            step_id=search_step.step_id,
            action=ToolName.SEARCH_PUBMED.value,
            summary=f"Retrieved {len(papers)} PubMed record(s).",
            provenance=tuple(search_result.provenance),
            retrieved_papers=papers,
        )
        trace_recorder.record(
            TraceEventType.TOOL_CALL,
            component="pubmed",
            operation="search_pubmed",
            status=TraceStatus.SUCCESS,
            summary=f"Retrieved {len(papers)} PubMed record(s).",
            step_id=search_step.step_id,
            tool=ToolName.SEARCH_PUBMED.value,
            provenance=tuple(search_result.provenance),
        )

        extracted: list[EvidenceRecord] = []
        extraction_provenance: list[Provenance] = []
        extraction_warnings: list[str] = []
        for paper in papers:
            if not paper.abstract:
                extraction_warnings.append(f"Skipped PMID {paper.pmid}: no abstract was available.")
                continue
            extraction = extract_literature_evidence(
                EvidenceExtractionRequest(
                    source=LiteratureSource(
                        pmid=paper.pmid,
                        cancer=resolution.canonical_name,
                        title=paper.title,
                        text=paper.abstract,
                        query=paper.query,
                    )
                ),
                provider=self._evidence_provider,
            )
            if extraction.status == ResultStatus.FAILURE or extraction.data is None:
                return self._failure(
                    "evidence_extraction_failed",
                    extraction.error.message if extraction.error else "Evidence extraction failed.",
                    (*state.provenance, *extraction.provenance),
                )
            extracted.extend(extraction.data.records)
            extraction_provenance.extend(extraction.provenance)
            extraction_warnings.extend(extraction.warnings)
            for call_index in range(extraction.data.provider_calls):
                trace_recorder.record(
                    TraceEventType.MODEL_CALL,
                    component="evidence_extractor",
                    operation="extract_passage",
                    status=TraceStatus.SUCCESS,
                    summary=f"Completed bounded evidence extraction call {call_index + 1} for PMID {paper.pmid}.",
                    step_id="extract_literature_evidence",
                )
            trace_recorder.record(
                TraceEventType.TOOL_CALL,
                component="evidence",
                operation="extract_literature_evidence",
                status=TraceStatus.SUCCESS,
                summary=f"Extracted {len(extraction.data.records)} record(s) from PMID {paper.pmid}.",
                step_id="extract_literature_evidence",
                tool=ToolName.EXTRACT_LITERATURE_EVIDENCE.value,
                provenance=tuple(extraction.provenance),
            )
        evidence = tuple(extracted)
        candidates = tuple(dict.fromkeys(record.lncrna for record in evidence))
        state = self._record_stage(
            state,
            step_id="extract_literature_evidence",
            action=ToolName.EXTRACT_LITERATURE_EVIDENCE.value,
            summary=f"Extracted {len(evidence)} source-grounded evidence record(s).",
            provenance=tuple(extraction_provenance),
            literature_evidence=evidence,
            candidate_lncrnas=candidates,
            warnings=(*state.warnings, *extraction_warnings),
        )
        if not evidence:
            return self._failure(
                "no_literature_evidence",
                "No source-grounded lncRNA evidence was extracted.",
                state.provenance,
            )

        agent_run = AgentLoop(
            registry=registry,
            replanning_provider=self._replanning_provider,
        ).run(state, analysis_plan)
        state = agent_run.state
        self._record_agent_events(trace_recorder, agent_run.events, state)
        if state.status == ResearchStatus.FAILED:
            return self._failure("analysis_workflow_failed", state.warnings[-1], state.provenance)
        if not state.survival_results and not state.de_results:
            detail = f" Last workflow warning: {state.warnings[-1]}" if state.warnings else ""
            return self._failure(
                "no_valid_analysis_completed",
                "No valid statistical analysis completed after capability inspection." + detail,
                state.provenance,
            )

        claims, references, grounded_claims = self._verification_inputs(state, evidence)
        verification_result = verify_claims(
            VerificationRequest(claims=tuple(claims), evidence=tuple(references)),
            semantic_provider=self._semantic_verifier_provider,
        )
        if verification_result.status == ResultStatus.FAILURE or verification_result.data is None:
            return self._failure(
                "verification_failed",
                verification_result.error.message if verification_result.error else "Verification failed.",
                (*state.provenance, *verification_result.provenance),
            )
        verification = verification_result.data
        if not verification.passed:
            return self._failure(
                "claims_not_verified",
                "The final claims did not pass verification.",
                (*state.provenance, *verification_result.provenance),
            )
        if verification.semantic_checks:
            trace_recorder.record(
                TraceEventType.MODEL_CALL,
                component="verifier",
                operation="semantic_claim_support",
                status=TraceStatus.SUCCESS,
                summary=f"Semantically checked {len(verification.semantic_checks)} claim(s).",
            )
        trace_recorder.record(
            TraceEventType.VERIFICATION,
            component="verifier",
            operation="verify_claims",
            status=TraceStatus.SUCCESS,
            summary=f"Verified {len(claims)} final claim(s).",
            step_id="verify_claims",
            tool="verify_claims",
            provenance=tuple(verification_result.provenance),
        )

        limitations = tuple(
            dict.fromkeys(
                (
                    *state.warnings,
                    *state.proposed_alternatives,
                    "Statistical associations are cohort-specific and do not establish causation.",
                    "This workflow prioritizes research evidence and does not clinically validate a biomarker.",
                )
            )
        )
        final_result = GroundedFinalResult(
            research_question=request.research_question,
            canonical_cancer_name=resolution.canonical_name,
            tcga_project_id=resolution.tcga_project_id,
            candidate_lncrnas=candidates,
            claims=tuple(grounded_claims),
            limitations=limitations,
            verification_passed=True,
        )
        report = self._render_report(final_result)
        state = state.updated(
            verification_report=verification,
            completed_steps=(*state.completed_steps, "verify_claims", "finalize_result"),
            observations=(
                *state.observations,
                Observation(
                    action="verify_claims",
                    summary=f"Verified {len(claims)} grounded claim(s).",
                    provenance=tuple(verification_result.provenance),
                ),
            ),
            provenance=(*state.provenance, *verification_result.provenance),
            research_plan=(),
            current_step=None,
            next_action=None,
            status=ResearchStatus.COMPLETED,
            final_report=report,
        )
        trace = trace_recorder.finish(
            status=TraceStatus.SUCCESS,
            summary="Vertical-slice execution completed with verified claims.",
        )
        trace_path = (
            save_trace(self._trace_directory, trace)
            if self._trace_directory is not None
            else None
        )
        data = VerticalSliceResult(
            state=state,
            initial_plan=plan,
            executed_plan=agent_run.plan,
            events=agent_run.events,
            retrieved_papers=papers,
            literature_evidence=evidence,
            survival_results=state.survival_results,
            de_results=state.de_results,
            verification=verification,
            final_result=final_result,
            replan_count=agent_run.replan_count,
            trace=trace,
            trace_path=trace_path,
        )
        return ToolResult[VerticalSliceResult](
            status=ResultStatus.SUCCESS,
            data=data,
            warnings=list(limitations),
            provenance=list(state.provenance),
        )

    @staticmethod
    def _record_agent_events(
        recorder: TraceRecorder,
        events: tuple[AgentEvent, ...],
        state: ResearchState,
    ) -> None:
        event_types = {
            AgentEventType.ACTION_SELECTED: TraceEventType.ACTION,
            AgentEventType.TOOL_EXECUTED: TraceEventType.TOOL_CALL,
            AgentEventType.OBSERVATION_RECORDED: TraceEventType.OBSERVATION,
            AgentEventType.REPLAN_REQUESTED: TraceEventType.REPLAN,
            AgentEventType.REPLAN_REPAIRED: TraceEventType.REPLAN,
            AgentEventType.PLAN_REVISED: TraceEventType.REPLAN,
            AgentEventType.ACTION_FAILED: TraceEventType.ACTION,
            AgentEventType.STOPPED: TraceEventType.ACTION,
        }
        for event in events:
            provenance = tuple(
                item
                for observation in state.observations
                if event.tool is not None and observation.action == event.tool.value
                for item in observation.provenance
            )
            recorder.record(
                event_types[event.event_type],
                component="agent_loop",
                operation=event.event_type.value,
                status=(
                    TraceStatus.FAILURE
                    if event.event_type == AgentEventType.ACTION_FAILED
                    else TraceStatus.SUCCESS
                ),
                summary=event.summary,
                step_id=event.step_id,
                tool=event.tool.value if event.tool else None,
                provenance=provenance,
            )
            if event.event_type == AgentEventType.REPLAN_REQUESTED:
                recorder.record(
                    TraceEventType.MODEL_CALL,
                    component="replanner",
                    operation="revise_plan",
                    status=TraceStatus.SUCCESS,
                    summary="Requested a bounded plan revision from the replanning provider.",
                    step_id=event.step_id,
                )
            if event.event_type == AgentEventType.REPLAN_REPAIRED:
                recorder.record(
                    TraceEventType.MODEL_CALL,
                    component="replanner",
                    operation="repair_plan_revision",
                    status=TraceStatus.SUCCESS,
                    summary="Made one bounded repair call after rejecting an invalid revision.",
                    step_id=event.step_id,
                )
    def _build_registry(self, request: VerticalSliceRequest) -> ToolRegistry:
        def run_survival(value: AnalysisInputReference) -> ToolResult:
            prepared = request.survival_inputs.get(value.input_id)
            if prepared is not None:
                return run_survival_analysis(prepared)
            source = request.tcga_survival_inputs[value.input_id]
            cohort_result = self._tcga_survival_client.prepare(source)
            if cohort_result.status == ResultStatus.FAILURE or cohort_result.data is None:
                return cohort_result
            return run_survival_analysis(cohort_result.data.survival_request())

        return ToolRegistry(
            (
                ToolSpec(
                    ToolName.SEARCH_PUBMED,
                    "Search PubMed and return normalized citation records.",
                    PubMedSearchRequest,
                    lambda value: search_pubmed(value, client=self._pubmed_client),
                ),
                ToolSpec(
                    ToolName.INSPECT_TCGA_CAPABILITIES,
                    "Inspect whether a TCGA project supports DE and survival analyses.",
                    TCCapabilityRequest,
                    lambda value: inspect_tcga_capabilities(value, client=self._gdc_client),
                ),
                ToolSpec(
                    ToolName.RUN_SURVIVAL_ANALYSIS,
                    "Prepare supported TCGA inputs when requested, then run deterministic survival analysis.",
                    AnalysisInputReference,
                    run_survival,
                ),
                ToolSpec(
                    ToolName.RUN_DIFFERENTIAL_EXPRESSION,
                    "Run deterministic differential expression using a prepared input reference.",
                    AnalysisInputReference,
                    lambda value: run_differential_expression(
                        request.differential_expression_inputs[value.input_id]
                    ),
                ),
            )
        )

    @staticmethod
    def _validate_and_split_plan(
        registry: ToolRegistry,
        request: VerticalSliceRequest,
        plan: ResearchPlan,
        canonical_cancer: str,
        project_id: str,
    ) -> tuple[PlanStep, ResearchPlan]:
        allowed = {
            ToolName.SEARCH_PUBMED,
            ToolName.INSPECT_TCGA_CAPABILITIES,
            ToolName.RUN_DIFFERENTIAL_EXPRESSION,
            ToolName.RUN_SURVIVAL_ANALYSIS,
        }
        if any(step.tool not in allowed for step in plan.steps):
            raise ValueError("vertical-slice plans may contain only retrieval, capability, and analysis tools")
        searches = [step for step in plan.steps if step.tool == ToolName.SEARCH_PUBMED]
        capabilities = [step for step in plan.steps if step.tool == ToolName.INSPECT_TCGA_CAPABILITIES]
        analyses = [
            step
            for step in plan.steps
            if step.tool in {ToolName.RUN_DIFFERENTIAL_EXPRESSION, ToolName.RUN_SURVIVAL_ANALYSIS}
        ]
        if len(searches) != 1 or len(capabilities) != 1 or not analyses:
            raise ValueError("plan requires one PubMed search, one capability check, and at least one analysis")
        search_index = plan.steps.index(searches[0])
        capability_index = plan.steps.index(capabilities[0])
        if search_index > capability_index or any(plan.steps.index(step) < capability_index for step in analyses):
            raise ValueError("plan must retrieve literature before capability checking and analysis")
        search_request = registry.validate_step(searches[0])
        if search_request.cancer.casefold() != canonical_cancer.casefold():
            raise ValueError("PubMed cancer argument must match the resolved cancer")
        capability_request = registry.validate_step(capabilities[0])
        if capability_request.project_id != project_id:
            raise ValueError("TCGA capability argument must match the resolved project")
        for step in analyses:
            parsed = registry.validate_step(step)
            inputs = (
                {**request.survival_inputs, **request.tcga_survival_inputs}
                if step.tool == ToolName.RUN_SURVIVAL_ANALYSIS
                else request.differential_expression_inputs
            )
            analysis_request = inputs.get(parsed.input_id)
            if analysis_request is None:
                raise ValueError(f"unknown {step.tool.value} input reference: {parsed.input_id}")
            if analysis_request.project_id != project_id:
                raise ValueError("analysis input must match the resolved TCGA project")
        analysis_steps = tuple(step for step in plan.steps if step is not searches[0])
        return searches[0], ResearchPlan.model_validate({**plan.model_dump(), "steps": analysis_steps})

    @staticmethod
    def _record_stage(
        state: ResearchState,
        *,
        step_id: str,
        action: str,
        summary: str,
        provenance: tuple[Provenance, ...],
        **changes: Any,
    ) -> ResearchState:
        return state.updated(
            **changes,
            completed_steps=(*state.completed_steps, step_id),
            observations=(
                *state.observations,
                Observation(action=action, summary=summary, provenance=provenance),
            ),
            provenance=(*state.provenance, *provenance),
        )

    @staticmethod
    def _verification_inputs(
        state: ResearchState, evidence: tuple[EvidenceRecord, ...]
    ) -> tuple[list[VerificationClaim], list[EvidenceReference], list[GroundedClaim]]:
        claims: list[VerificationClaim] = []
        references: list[EvidenceReference] = []
        grounded: list[GroundedClaim] = []
        for index, record in enumerate(evidence, start=1):
            claim_id = f"literature-{index}"
            reference_id = f"pubmed:{record.pmid}:{record.passage_id}"
            provenance = (Provenance(source="NCBI PubMed", source_id=record.pmid),)
            numeric = {"cohort_size": float(record.cohort_size)} if record.cohort_size else {}
            references.append(
                EvidenceReference(
                    reference_id=reference_id,
                    source_type="PubMed",
                    source_text=record.supporting_text,
                    gene_identifier=record.lncrna,
                    pmid=record.pmid,
                    numeric_values=numeric,
                    provenance=provenance,
                )
            )
            assertions = (
                NumericAssertion(
                    evidence_reference_id=reference_id,
                    metric="cohort_size",
                    reported_value=float(record.cohort_size),
                ),
            ) if record.cohort_size else ()
            claims.append(
                VerificationClaim(
                    claim_id=claim_id,
                    text=record.supporting_text,
                    kind=ClaimKind.LITERATURE,
                    gene_identifier=record.lncrna,
                    pmid=record.pmid,
                    evidence_reference_ids=(reference_id,),
                    numeric_assertions=assertions,
                )
            )
            grounded.append(
                GroundedClaim(
                    claim_id=claim_id,
                    text=record.supporting_text,
                    source_ids=(reference_id,),
                    pmid=record.pmid,
                    numeric_values=numeric,
                )
            )
        for index, result in enumerate(state.survival_results, start=1):
            claim_id = f"survival-{index}"
            reference_id = f"survival:{result.project_id}:{result.gene}"
            text = (
                f"In {result.project_id}, {result.gene} high versus low expression was associated "
                f"with {result.configuration.endpoint.value} (n={result.sample_count}, "
                f"events={result.event_count}, "
                f"HR={result.hazard_ratio:.6g}, 95% CI {result.confidence_interval_lower:.6g}–"
                f"{result.confidence_interval_upper:.6g}, Cox p={result.cox_p_value:.6g}, "
                f"log-rank p={result.log_rank_p_value:.6g})."
            )
            numeric = {
                "sample_count": float(result.sample_count),
                "event_count": float(result.event_count),
                "hazard_ratio": result.hazard_ratio,
                "confidence_interval_lower": result.confidence_interval_lower,
                "confidence_interval_upper": result.confidence_interval_upper,
                "cox_p_value": result.cox_p_value,
                "log_rank_p_value": result.log_rank_p_value,
            }
            provenance = tuple(
                item
                for observation in state.observations
                if observation.action == ToolName.RUN_SURVIVAL_ANALYSIS.value
                for item in observation.provenance
            )
            references.append(
                EvidenceReference(
                    reference_id=reference_id,
                    source_type="deterministic survival analysis",
                    source_text=text,
                    gene_identifier=result.gene,
                    numeric_values=numeric,
                    provenance=provenance,
                )
            )
            assertions = tuple(
                NumericAssertion(
                    evidence_reference_id=reference_id, metric=metric, reported_value=value
                )
                for metric, value in numeric.items()
            )
            claims.append(
                VerificationClaim(
                    claim_id=claim_id,
                    text=text,
                    kind=ClaimKind.SURVIVAL,
                    gene_identifier=result.gene,
                    evidence_reference_ids=(reference_id,),
                    numeric_assertions=assertions,
                )
            )
            grounded.append(
                GroundedClaim(
                    claim_id=claim_id,
                    text=text,
                    source_ids=(reference_id,),
                    numeric_values=numeric,
                )
            )
        for result_index, result in enumerate(state.de_results, start=1):
            provenance = tuple(
                item
                for observation in state.observations
                if observation.action == ToolName.RUN_DIFFERENTIAL_EXPRESSION.value
                for item in observation.provenance
            )
            for feature_index, feature in enumerate(result.results, start=1):
                gene = feature.gene_symbol or feature.feature_id
                claim_id = f"de-{result_index}-{feature_index}"
                reference_id = f"de:{result.project_id}:{feature.feature_id}"
                text = (
                    f"In {result.project_id}, {gene} had tumor-versus-normal log2 fold change "
                    f"{feature.log2_fold_change:.6g} (p={feature.p_value:.6g}, adjusted "
                    f"p={feature.adjusted_p_value:.6g}; {result.case_sample_count} cases and "
                    f"{result.control_sample_count} controls)."
                )
                numeric = {
                    "log2_fold_change": feature.log2_fold_change,
                    "p_value": feature.p_value,
                    "adjusted_p_value": feature.adjusted_p_value,
                    "case_sample_count": float(result.case_sample_count),
                    "control_sample_count": float(result.control_sample_count),
                }
                references.append(
                    EvidenceReference(
                        reference_id=reference_id,
                        source_type="deterministic differential-expression analysis",
                        source_text=text,
                        gene_identifier=gene,
                        numeric_values=numeric,
                        provenance=provenance,
                    )
                )
                assertions = tuple(
                    NumericAssertion(
                        evidence_reference_id=reference_id, metric=metric, reported_value=value
                    )
                    for metric, value in numeric.items()
                )
                claims.append(
                    VerificationClaim(
                        claim_id=claim_id,
                        text=text,
                        kind=ClaimKind.DIFFERENTIAL_EXPRESSION,
                        gene_identifier=gene,
                        evidence_reference_ids=(reference_id,),
                        numeric_assertions=assertions,
                    )
                )
                grounded.append(
                    GroundedClaim(
                        claim_id=claim_id,
                        text=text,
                        source_ids=(reference_id,),
                        numeric_values=numeric,
                    )
                )
        return claims, references, grounded

    @staticmethod
    def _render_report(result: GroundedFinalResult) -> str:
        lines = [
            f"Question: {result.research_question}",
            f"Resolved cancer: {result.canonical_cancer_name} ({result.tcga_project_id})",
            "Verified claims:",
        ]
        for claim in result.claims:
            citation = f" PMID: {claim.pmid}." if claim.pmid else ""
            lines.append(
                f"- {claim.text}{citation} Sources: {', '.join(claim.source_ids)}"
            )
        lines.append("Limitations:")
        lines.extend(f"- {limitation}" for limitation in result.limitations)
        lines.append(result.disclaimer)
        return "\n".join(lines)

    @staticmethod
    def _result_error(result: Any) -> str:
        if isinstance(result, ToolResult) and result.error is not None:
            return result.error.message
        return "Tool returned an invalid or failed result."

    @staticmethod
    def _failure(code: str, message: str, provenance: Any) -> ToolResult[VerticalSliceResult]:
        return ToolResult[VerticalSliceResult](
            status=ResultStatus.FAILURE,
            error=ToolError(code=code, message=message, retryable=False),
            provenance=list(provenance),
        )
