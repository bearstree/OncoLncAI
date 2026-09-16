"""Deterministic grading for the versioned OncoLnc-AgentBench suite."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oncolncai.tracing import TraceEventType


class BenchmarkCaseType(StrEnum):
    STANDARD = "standard"
    REPLANNING = "replanning"
    FAILURE = "failure"
    AMBIGUITY = "ambiguity"


class ExpectedRunStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_INPUT = "needs_input"


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    case_type: BenchmarkCaseType
    question: str = Field(min_length=1)
    cancer_name: str = Field(min_length=1)
    expected_cancer_status: str = Field(min_length=1)
    expected_project_id: str | None = None
    expected_run_status: ExpectedRunStatus
    required_operations: tuple[str, ...] = ()
    prohibited_operations: tuple[str, ...] = ()
    required_skipped_steps: tuple[str, ...] = ()
    expected_replan_count: int = Field(default=0, ge=0)
    expected_error_code: str | None = None
    require_verification: bool = False
    require_provenance: bool = True
    max_model_calls: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_expectations(self) -> BenchmarkCase:
        if set(self.required_operations) & set(self.prohibited_operations):
            raise ValueError("an operation cannot be both required and prohibited")
        if self.case_type == BenchmarkCaseType.REPLANNING and self.expected_replan_count < 1:
            raise ValueError("replanning cases must expect at least one replan")
        if self.case_type == BenchmarkCaseType.FAILURE and not self.expected_error_code:
            raise ValueError("failure cases require an expected_error_code")
        if self.expected_run_status == ExpectedRunStatus.COMPLETED and self.expected_error_code:
            raise ValueError("completed cases cannot expect an error")
        return self


class BenchmarkSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_name: str = Field(pattern=r"^OncoLnc-AgentBench$")
    version: str = Field(pattern=r"^1\.0\.0$")
    cases: tuple[BenchmarkCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_ids(self) -> BenchmarkSuite:
        identifiers = [case.case_id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("benchmark case IDs must be unique")
        return self


class BenchmarkObservation(BaseModel):
    """Compact observed behavior produced from a run or trace adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    cancer_status: str
    project_id: str | None = None
    run_status: ExpectedRunStatus
    operations: tuple[str, ...] = ()
    skipped_steps: tuple[str, ...] = ()
    replan_count: int = Field(default=0, ge=0)
    verification_passed: bool | None = None
    provenance_source_ids: tuple[str, ...] = ()
    error_code: str | None = None
    model_call_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)


class BenchmarkObservations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_version: str = Field(pattern=r"^1\.0\.0$")
    observations: tuple[BenchmarkObservation, ...]

    @model_validator(mode="after")
    def validate_observation_ids(self) -> BenchmarkObservations:
        identifiers = [observation.case_id for observation in self.observations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("benchmark observation case IDs must be unique")
        return self


class BenchmarkCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    expected: str
    observed: str


class BenchmarkCaseGrade(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    case_type: BenchmarkCaseType
    passed: bool
    checks: tuple[BenchmarkCheck, ...]
    unnecessary_tool_calls: int = Field(ge=0)


class AgentMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_case_pass_rate: float = Field(ge=0, le=1)
    cancer_resolution_accuracy: float = Field(ge=0, le=1)
    workflow_outcome_accuracy: float = Field(ge=0, le=1)
    tool_selection_accuracy: float = Field(ge=0, le=1)
    replanning_success_rate: float = Field(ge=0, le=1)
    failure_handling_accuracy: float = Field(ge=0, le=1)
    ambiguity_handling_accuracy: float = Field(ge=0, le=1)
    verification_accuracy: float = Field(ge=0, le=1)
    provenance_retention_rate: float = Field(ge=0, le=1)


class EfficiencyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_model_calls: int = Field(ge=0)
    total_tool_calls: int = Field(ge=0)
    average_model_calls_per_case: float = Field(ge=0)
    average_tool_calls_per_case: float = Field(ge=0)
    call_budget_compliance_rate: float = Field(ge=0, le=1)
    unnecessary_tool_call_rate: float = Field(ge=0, le=1)


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_name: str
    benchmark_version: str
    case_count: int = Field(ge=1)
    passed_case_count: int = Field(ge=0)
    case_grades: tuple[BenchmarkCaseGrade, ...]
    agent_metrics: AgentMetrics
    efficiency_metrics: EfficiencyMetrics


def load_benchmark_suite(path: Path) -> BenchmarkSuite:
    return BenchmarkSuite.model_validate_json(path.read_text(encoding="utf-8"))


def load_benchmark_observations(path: Path) -> BenchmarkObservations:
    return BenchmarkObservations.model_validate_json(path.read_text(encoding="utf-8"))


def observation_from_vertical_result(case_id: str, result: object) -> BenchmarkObservation:
    """Adapt a successful vertical-slice result without copying raw scientific data."""

    from oncolncai.vertical_slice import VerticalSliceResult

    validated = VerticalSliceResult.model_validate(result)
    operations = tuple(
        dict.fromkeys(
            event.tool or event.operation
            for event in validated.trace.events
            if event.event_type in {TraceEventType.TOOL_CALL, TraceEventType.VERIFICATION}
        )
    )
    provenance_ids = tuple(
        dict.fromkeys(
            item.source_id
            for item in validated.state.provenance
            if item.source_id is not None
        )
    )
    return BenchmarkObservation(
        case_id=case_id,
        cancer_status="resolved",
        project_id=validated.state.tcga_project_id,
        run_status=ExpectedRunStatus.COMPLETED,
        operations=operations,
        skipped_steps=validated.state.skipped_steps,
        replan_count=validated.trace.replan_count,
        verification_passed=validated.verification.passed,
        provenance_source_ids=provenance_ids,
        model_call_count=validated.trace.model_call_count,
        tool_call_count=validated.trace.tool_call_count,
    )


def _check(name: str, passed: bool, expected: object, observed: object) -> BenchmarkCheck:
    return BenchmarkCheck(
        name=name,
        passed=passed,
        expected=str(expected),
        observed=str(observed),
    )


def grade_case(case: BenchmarkCase, observed: BenchmarkObservation) -> BenchmarkCaseGrade:
    if case.case_id != observed.case_id:
        raise ValueError("case and observation IDs must match")
    operations = set(observed.operations)
    required = set(case.required_operations)
    prohibited = set(case.prohibited_operations)
    skipped = set(observed.skipped_steps)
    checks = (
        _check(
            "cancer_resolution",
            observed.cancer_status == case.expected_cancer_status
            and observed.project_id == case.expected_project_id,
            (case.expected_cancer_status, case.expected_project_id),
            (observed.cancer_status, observed.project_id),
        ),
        _check(
            "workflow_outcome",
            observed.run_status == case.expected_run_status,
            case.expected_run_status.value,
            observed.run_status.value,
        ),
        _check(
            "tool_selection",
            required <= operations and not prohibited & operations,
            f"required={sorted(required)}, prohibited={sorted(prohibited)}",
            sorted(operations),
        ),
        _check(
            "skipped_steps",
            set(case.required_skipped_steps) <= skipped,
            sorted(case.required_skipped_steps),
            sorted(skipped),
        ),
        _check(
            "replanning",
            observed.replan_count == case.expected_replan_count,
            case.expected_replan_count,
            observed.replan_count,
        ),
        _check(
            "error_handling",
            observed.error_code == case.expected_error_code,
            case.expected_error_code,
            observed.error_code,
        ),
        _check(
            "verification",
            not case.require_verification or observed.verification_passed is True,
            True if case.require_verification else "not required",
            observed.verification_passed,
        ),
        _check(
            "provenance",
            not case.require_provenance or bool(observed.provenance_source_ids),
            "present" if case.require_provenance else "not required",
            "present" if observed.provenance_source_ids else "missing",
        ),
        _check(
            "call_budget",
            observed.model_call_count <= case.max_model_calls
            and observed.tool_call_count <= case.max_tool_calls,
            f"model<={case.max_model_calls}, tool<={case.max_tool_calls}",
            f"model={observed.model_call_count}, tool={observed.tool_call_count}",
        ),
    )
    unnecessary = len(operations - required)
    return BenchmarkCaseGrade(
        case_id=case.case_id,
        case_type=case.case_type,
        passed=all(check.passed for check in checks),
        checks=checks,
        unnecessary_tool_calls=unnecessary,
    )


def _rate(grades: tuple[BenchmarkCaseGrade, ...], check_name: str) -> float:
    checks = [check for grade in grades for check in grade.checks if check.name == check_name]
    return sum(check.passed for check in checks) / len(checks) if checks else 1.0


def _typed_rate(
    grades: tuple[BenchmarkCaseGrade, ...], case_type: BenchmarkCaseType
) -> float:
    selected = [grade for grade in grades if grade.case_type == case_type]
    return sum(grade.passed for grade in selected) / len(selected) if selected else 1.0


def run_benchmark(
    suite: BenchmarkSuite,
    observations: BenchmarkObservations,
) -> BenchmarkReport:
    """Grade one complete observation set with no model-based judging."""

    if observations.benchmark_version != suite.version:
        raise ValueError("benchmark and observation versions must match")
    observed_by_id = {observation.case_id: observation for observation in observations.observations}
    expected_ids = {case.case_id for case in suite.cases}
    if set(observed_by_id) != expected_ids:
        missing = sorted(expected_ids - set(observed_by_id))
        extra = sorted(set(observed_by_id) - expected_ids)
        raise ValueError(f"observations must match suite cases; missing={missing}, extra={extra}")
    grades = tuple(grade_case(case, observed_by_id[case.case_id]) for case in suite.cases)
    total_model = sum(item.model_call_count for item in observations.observations)
    total_tool = sum(item.tool_call_count for item in observations.observations)
    unnecessary = sum(grade.unnecessary_tool_calls for grade in grades)
    budget_checks = [
        check for grade in grades for check in grade.checks if check.name == "call_budget"
    ]
    verification_checks = [
        next(check for check in grade.checks if check.name == "verification")
        for grade, case in zip(grades, suite.cases)
        if case.require_verification
    ]
    count = len(suite.cases)
    return BenchmarkReport(
        benchmark_name=suite.benchmark_name,
        benchmark_version=suite.version,
        case_count=count,
        passed_case_count=sum(grade.passed for grade in grades),
        case_grades=grades,
        agent_metrics=AgentMetrics(
            overall_case_pass_rate=sum(grade.passed for grade in grades) / count,
            cancer_resolution_accuracy=_rate(grades, "cancer_resolution"),
            workflow_outcome_accuracy=_rate(grades, "workflow_outcome"),
            tool_selection_accuracy=_rate(grades, "tool_selection"),
            replanning_success_rate=_typed_rate(grades, BenchmarkCaseType.REPLANNING),
            failure_handling_accuracy=_typed_rate(grades, BenchmarkCaseType.FAILURE),
            ambiguity_handling_accuracy=_typed_rate(grades, BenchmarkCaseType.AMBIGUITY),
            verification_accuracy=(
                sum(check.passed for check in verification_checks) / len(verification_checks)
                if verification_checks else 1.0
            ),
            provenance_retention_rate=_rate(grades, "provenance"),
        ),
        efficiency_metrics=EfficiencyMetrics(
            total_model_calls=total_model,
            total_tool_calls=total_tool,
            average_model_calls_per_case=total_model / count,
            average_tool_calls_per_case=total_tool / count,
            call_budget_compliance_rate=sum(check.passed for check in budget_checks) / count,
            unnecessary_tool_call_rate=unnecessary / total_tool if total_tool else 0.0,
        ),
    )
