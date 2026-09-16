"""Bounded planning and allow-listed tool registration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from oncolncai.cancer_resolver import CancerResolveRequest, resolve_cancer
from oncolncai.differential_expression import DifferentialExpressionRequest, run_differential_expression
from oncolncai.evidence import EvidenceExtractionRequest, StructuredGenerationProvider, extract_literature_evidence
from oncolncai.evidence_fusion import EvidenceFusionRequest, prioritize_evidence
from oncolncai.geo import GEOClient, GEOSearchRequest, search_geo
from oncolncai.localization import LocalizationBackend, LocalizationRequest, predict_lncrna_localization
from oncolncai.providers import LLMProvider
from oncolncai.pubmed import PubMedClient, PubMedSearchRequest, search_pubmed
from oncolncai.schemas import ResultStatus, ToolError, ToolResult
from oncolncai.survival import SurvivalRequest, run_survival_analysis
from oncolncai.tcga_capabilities import GDCCapabilityClient, TCCapabilityRequest, inspect_tcga_capabilities


class ResearchIntent(StrEnum):
    DISCOVER_BIOMARKERS = "discover_biomarkers"
    VALIDATE_LNCRNA = "validate_lncrna"
    DIFFERENTIAL_EXPRESSION = "differential_expression"
    PROGNOSTIC_ANALYSIS = "prognostic_analysis"
    LITERATURE_EVIDENCE = "literature_evidence"


class ToolName(StrEnum):
    RESOLVE_CANCER = "resolve_cancer"
    SEARCH_PUBMED = "search_pubmed"
    EXTRACT_LITERATURE_EVIDENCE = "extract_literature_evidence"
    INSPECT_TCGA_CAPABILITIES = "inspect_tcga_capabilities"
    RUN_DIFFERENTIAL_EXPRESSION = "run_differential_expression"
    RUN_SURVIVAL_ANALYSIS = "run_survival_analysis"
    SEARCH_GEO = "search_geo"
    PRIORITIZE_EVIDENCE = "prioritize_evidence"
    PREDICT_LNCRNA_LOCALIZATION = "predict_lncrna_localization"


class PlanningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    cancer_name: str | None = Field(default=None, min_length=1, max_length=200)
    tcga_project_id: str | None = Field(default=None, pattern=r"^TCGA-[A-Z0-9]+$")
    intent_hint: ResearchIntent | None = None


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    step_id: str = Field(min_length=1, max_length=60, pattern=r"^[a-z][a-z0-9_]*$")
    tool: ToolName
    purpose: str = Field(min_length=1, max_length=240)
    arguments: dict[str, JsonValue]


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    intent: ResearchIntent
    goal: str = Field(min_length=1, max_length=300)
    steps: tuple[PlanStep, ...] = Field(min_length=1, max_length=8)
    assumptions: tuple[str, ...] = Field(default=(), max_length=5)
    requires_user_input: bool = False

    @model_validator(mode="after")
    def validate_step_ids(self) -> ResearchPlan:
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("plan step IDs must be unique")
        return self


ToolHandler = Callable[[BaseModel], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: ToolName
    description: str
    request_model: type[BaseModel]
    handler: ToolHandler | None = None


class ToolRegistry:
    """Allow-list of typed tools available to a planner and executor."""

    def __init__(self, specs: tuple[ToolSpec, ...] = ()) -> None:
        self._specs: dict[ToolName, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name.value}")
        self._specs[spec.name] = spec

    @property
    def tool_names(self) -> tuple[ToolName, ...]:
        return tuple(self._specs)

    def describe(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "name": spec.name.value,
                "description": spec.description,
                "input_schema": spec.request_model.model_json_schema(),
            }
            for spec in self._specs.values()
        )

    def validate_step(self, step: PlanStep) -> BaseModel:
        spec = self._specs.get(step.tool)
        if spec is None:
            raise ValueError(f"tool is not registered: {step.tool.value}")
        return spec.request_model.model_validate(step.arguments)

    def validate_plan(self, plan: ResearchPlan) -> None:
        for step in plan.steps:
            self.validate_step(step)

    def is_executable(self, tool_name: ToolName) -> bool:
        spec = self._specs.get(tool_name)
        return spec is not None and spec.handler is not None

    def execute(self, tool_name: ToolName | str, arguments: dict[str, JsonValue]) -> Any:
        try:
            name = ToolName(tool_name)
        except ValueError as exc:
            raise ValueError(f"tool is not allowed: {tool_name}") from exc
        spec = self._specs.get(name)
        if spec is None:
            raise ValueError(f"tool is not registered: {name.value}")
        request = spec.request_model.model_validate(arguments)
        if spec.handler is None:
            raise RuntimeError(f"tool has no configured handler: {name.value}")
        return spec.handler(request)


def build_default_registry(
    *,
    pubmed_client: PubMedClient | None = None,
    gdc_client: GDCCapabilityClient | None = None,
    evidence_provider: StructuredGenerationProvider | None = None,
    geo_client: GEOClient | None = None,
    localization_backend: LocalizationBackend | None = None,
) -> ToolRegistry:
    """Register implemented tools, keeping optional handlers non-executable."""

    evidence_handler: ToolHandler | None = None
    if evidence_provider is not None:
        evidence_handler = lambda request: extract_literature_evidence(
            request, provider=evidence_provider
        )
    return ToolRegistry(
        (
            ToolSpec(
                ToolName.PREDICT_LNCRNA_LOCALIZATION,
                "Predict optional lncRNA localization as biological context only.",
                LocalizationRequest,
                None if localization_backend is None else lambda request: predict_lncrna_localization(request, backend=localization_backend),
            ),
            ToolSpec(
                ToolName.SEARCH_GEO,
                "Find and deterministically assess GEO validation datasets.",
                GEOSearchRequest,
                lambda request: search_geo(request, client=geo_client),
            ),
            ToolSpec(
                ToolName.PRIORITIZE_EVIDENCE,
                "Calculate a deterministic, component-level evidence prioritization score.",
                EvidenceFusionRequest,
                prioritize_evidence,
            ),
            ToolSpec(
                ToolName.RESOLVE_CANCER,
                "Resolve user cancer language to curated TCGA identifiers.",
                CancerResolveRequest,
                resolve_cancer,
            ),
            ToolSpec(
                ToolName.SEARCH_PUBMED,
                "Search PubMed and return normalized citation records.",
                PubMedSearchRequest,
                lambda request: search_pubmed(request, client=pubmed_client),
            ),
            ToolSpec(
                ToolName.EXTRACT_LITERATURE_EVIDENCE,
                "Extract source-grounded structured lncRNA evidence.",
                EvidenceExtractionRequest,
                evidence_handler,
            ),
            ToolSpec(
                ToolName.INSPECT_TCGA_CAPABILITIES,
                "Inspect whether a TCGA project supports DE and survival analyses.",
                TCCapabilityRequest,
                lambda request: inspect_tcga_capabilities(request, client=gdc_client),
            ),
            ToolSpec(
                ToolName.RUN_SURVIVAL_ANALYSIS,
                "Run deterministic Kaplan-Meier, log-rank, and Cox PH analysis.",
                SurvivalRequest,
                run_survival_analysis,
            ),
            ToolSpec(
                ToolName.RUN_DIFFERENTIAL_EXPRESSION,
                "Run deterministic CPM-normalized differential expression.",
                DifferentialExpressionRequest,
                run_differential_expression,
            ),
        )
    )


class Planner:
    """Request a bounded plan from an LLM provider and validate every action."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        registry: ToolRegistry,
        fallback_provider: LLMProvider | None = None,
        repair_attempts: int = 1,
    ) -> None:
        if repair_attempts not in (0, 1):
            raise ValueError("repair_attempts must be zero or one")
        self._provider = provider
        self._registry = registry
        self._fallback_provider = fallback_provider
        self._repair_attempts = repair_attempts

    def plan(self, request: PlanningRequest) -> ToolResult[ResearchPlan]:
        prompt = (
            "Create a concise executable research plan. Use only the listed tools. "
            "Do not include analysis tools that are not registered. The arguments for "
            "each step must be the direct JSON object accepted by that tool's input_schema; "
            "use exact property names and do not wrap values in description/value objects.\n"
            f"Request: {request.model_dump_json()}\n"
            f"Available tools: {self._registry.describe()}"
        )
        providers = [self._provider]
        if self._fallback_provider is not None:
            providers.append(self._fallback_provider)
        validation_error: str | None = None
        failure_code = "planner_provider_failed"
        attempts = 0
        for provider_index, provider in enumerate(providers):
            provider_attempts = 1 + (self._repair_attempts if provider_index == 0 else 0)
            for attempt_index in range(provider_attempts):
                attempts += 1
                attempt_prompt = prompt
                if validation_error is not None:
                    attempt_prompt += (
                        "\nThe previous proposal was rejected before execution. Correct only "
                        "the schema/consistency error and regenerate the complete plan. "
                        f"Validation error: {validation_error}"
                    )
                try:
                    raw_plan = provider.generate_structured(
                        prompt=attempt_prompt,
                        output_schema=ResearchPlan,
                    )
                    plan = ResearchPlan.model_validate(raw_plan)
                    self._registry.validate_plan(plan)
                    self._validate_context(plan, request)
                    warnings = []
                    if attempts > 1:
                        route = "fallback provider" if provider_index else "bounded repair"
                        warnings.append(f"Planner accepted after {route}; {attempts} model calls were made.")
                    return ToolResult[ResearchPlan](
                        status=ResultStatus.SUCCESS, data=plan, warnings=warnings
                    )
                except (ValidationError, ValueError) as exc:
                    validation_error = str(exc)
                    failure_code = "invalid_plan"
                    continue
                except Exception as exc:
                    if failure_code != "invalid_plan":
                        validation_error = str(exc) or type(exc).__name__
                        failure_code = "planner_provider_failed"
                    break
        return ToolResult[ResearchPlan](
            status=ResultStatus.FAILURE,
            error=ToolError(
                code=failure_code,
                message=validation_error or "planner provider failed",
                retryable=False,
            ),
            warnings=[f"Rejected {attempts} planner proposal(s) before tool execution."],
        )

    @staticmethod
    def _validate_context(plan: ResearchPlan, request: PlanningRequest) -> None:
        """Reject scientifically meaningful context mismatches; never repair them silently."""

        for step in plan.steps:
            project = step.arguments.get("project_id")
            if request.tcga_project_id and project is not None and project != request.tcga_project_id:
                raise ValueError(
                    f"step {step.step_id} project {project!r} does not match resolved "
                    f"project {request.tcga_project_id!r}"
                )
            cancer = step.arguments.get("cancer")
            if request.cancer_name and cancer is not None and str(cancer).casefold() != request.cancer_name.casefold():
                raise ValueError(
                    f"step {step.step_id} cancer {cancer!r} does not match resolved "
                    f"cancer {request.cancer_name!r}"
                )
