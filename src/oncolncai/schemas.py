"""Core contracts shared by deterministic tools and later agent components."""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResultStatus(StrEnum):
    """Outcome of a deterministic operation."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class Provenance(BaseModel):
    """Machine-readable origin of a result or claim."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    source_id: str | None = None
    query: str | None = None
    dataset_version: str | None = None


class ToolError(BaseModel):
    """Structured failure information safe to pass between components."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


ResultDataT = TypeVar("ResultDataT", bound=BaseModel)


class ToolResult(BaseModel, Generic[ResultDataT]):
    """Common envelope for auditable deterministic tool output."""

    model_config = ConfigDict(extra="forbid")

    status: ResultStatus
    data: ResultDataT | None = None
    provenance: list[Provenance] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: ToolError | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> ToolResult:
        if self.status == ResultStatus.FAILURE and self.error is None:
            raise ValueError("failure results must include an error")
        if self.status != ResultStatus.FAILURE and self.error is not None:
            raise ValueError("only failure results may include an error")
        if self.status == ResultStatus.SUCCESS and self.data is None:
            raise ValueError("successful results must include data")
        return self
