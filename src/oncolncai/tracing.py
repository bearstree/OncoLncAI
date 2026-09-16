"""Structured, persistence-safe execution tracing without reasoning content."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oncolncai.schemas import Provenance


class TraceEventType(StrEnum):
    RUN_STARTED = "run_started"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    ACTION = "action"
    OBSERVATION = "observation"
    REPLAN = "replan"
    VERIFICATION = "verification"
    RUN_COMPLETED = "run_completed"


class TraceStatus(StrEnum):
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"


class TraceEvent(BaseModel):
    """One auditable execution fact; prompts and raw payloads are deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    timestamp: datetime
    event_type: TraceEventType
    component: str = Field(min_length=1, max_length=100)
    operation: str = Field(min_length=1, max_length=100)
    status: TraceStatus
    summary: str = Field(min_length=1, max_length=500)
    step_id: str | None = None
    tool: str | None = None
    provenance: tuple[Provenance, ...] = ()


class ExecutionTrace(BaseModel):
    """Validated trace for one run, stored separately from scientific state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="1.0", pattern=r"^1\.0$")
    run_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    started_at: datetime
    completed_at: datetime
    status: TraceStatus
    events: tuple[TraceEvent, ...] = Field(min_length=2)
    tool_call_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    replan_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_trace(self) -> ExecutionTrace:
        if [event.sequence for event in self.events] != list(range(1, len(self.events) + 1)):
            raise ValueError("trace event sequences must be contiguous and start at 1")
        if self.events[0].event_type != TraceEventType.RUN_STARTED:
            raise ValueError("trace must start with run_started")
        if self.events[-1].event_type != TraceEventType.RUN_COMPLETED:
            raise ValueError("trace must end with run_completed")
        if self.events[-1].status != self.status:
            raise ValueError("final event status must match trace status")
        if self.completed_at < self.started_at:
            raise ValueError("trace completion cannot precede its start")
        if any(
            later.timestamp < earlier.timestamp
            for earlier, later in zip(self.events, self.events[1:])
        ):
            raise ValueError("trace timestamps must be nondecreasing")
        if self.tool_call_count != sum(
            event.event_type == TraceEventType.TOOL_CALL for event in self.events
        ):
            raise ValueError("tool_call_count does not match trace events")
        if self.model_call_count != sum(
            event.event_type == TraceEventType.MODEL_CALL for event in self.events
        ):
            raise ValueError("model_call_count does not match trace events")
        if self.replan_count != sum(
            event.event_type == TraceEventType.REPLAN and event.operation == "plan_revised"
            for event in self.events
        ):
            raise ValueError("replan_count does not match revised-plan events")
        return self


class TraceRecorder:
    """Append validated summaries and finalize them into an immutable trace."""

    def __init__(
        self,
        run_id: str,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._run_id = run_id
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._events: list[TraceEvent] = []
        self.record(
            TraceEventType.RUN_STARTED,
            component="vertical_slice",
            operation="run",
            status=TraceStatus.STARTED,
            summary="Vertical-slice execution started.",
        )

    def record(
        self,
        event_type: TraceEventType,
        *,
        component: str,
        operation: str,
        status: TraceStatus,
        summary: str,
        step_id: str | None = None,
        tool: str | None = None,
        provenance: tuple[Provenance, ...] = (),
    ) -> TraceEvent:
        event = TraceEvent(
            sequence=len(self._events) + 1,
            timestamp=self._clock(),
            event_type=event_type,
            component=component,
            operation=operation,
            status=status,
            summary=summary,
            step_id=step_id,
            tool=tool,
            provenance=provenance,
        )
        self._events.append(event)
        return event

    def finish(self, *, status: TraceStatus, summary: str) -> ExecutionTrace:
        if status == TraceStatus.STARTED:
            raise ValueError("a finished trace must have success or failure status")
        self.record(
            TraceEventType.RUN_COMPLETED,
            component="vertical_slice",
            operation="run",
            status=status,
            summary=summary,
        )
        return ExecutionTrace(
            run_id=self._run_id,
            started_at=self._events[0].timestamp,
            completed_at=self._events[-1].timestamp,
            status=status,
            events=tuple(self._events),
            tool_call_count=sum(
                event.event_type == TraceEventType.TOOL_CALL for event in self._events
            ),
            model_call_count=sum(
                event.event_type == TraceEventType.MODEL_CALL for event in self._events
            ),
            replan_count=sum(
                event.event_type == TraceEventType.REPLAN and event.operation == "plan_revised"
                for event in self._events
            ),
        )


def trace_filename(run_id: str) -> str:
    """Return a stable filename for an already schema-valid run ID."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run_id contains invalid filename characters")
    return f"{run_id}.trace.json"


def save_trace(directory: Path, trace: ExecutionTrace) -> Path:
    """Atomically save a trace without overwriting an existing run record."""

    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / trace_filename(trace.run_id)
    if destination.exists():
        raise FileExistsError(f"trace already exists: {destination}")
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{trace.run_id}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(trace.model_dump_json(indent=2))
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_trace(path: Path) -> ExecutionTrace:
    """Load and fully validate a persisted trace."""

    return ExecutionTrace.model_validate_json(path.read_text(encoding="utf-8"))
