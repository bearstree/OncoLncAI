"""Explicit research state and deterministic JSON checkpoint persistence."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oncolncai.differential_expression import DifferentialExpressionResult
from oncolncai.evidence import EvidenceRecord
from oncolncai.pubmed import PubMedRecord
from oncolncai.schemas import Provenance
from oncolncai.survival import SurvivalAnalysisResult
from oncolncai.tcga_capabilities import DataCapabilities
from oncolncai.verifier import VerificationReport


SCHEMA_VERSION = "1.0"
_CHECKPOINT_SLUG = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class ResearchStatus(StrEnum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"


class Observation(BaseModel):
    """Compact fact observed from an action, without hidden reasoning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    provenance: tuple[Provenance, ...] = ()


class Decision(BaseModel):
    """Auditable decision and the explicit evidence or rule supporting it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: str = Field(min_length=1)
    basis: str = Field(min_length=1)


class ResearchState(BaseModel):
    """Validated facts and workflow progress for one research run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    research_question: str = Field(min_length=1)
    user_cancer_name: str | None = None
    canonical_cancer_name: str | None = None
    cancer_synonyms: tuple[str, ...] = ()
    tcga_project_id: str | None = Field(default=None, pattern=r"^TCGA-[A-Z0-9]+$")
    intent: str | None = None
    initial_plan: tuple[str, ...] = ()
    research_plan: tuple[str, ...] = ()
    current_step: str | None = None
    next_action: str | None = None
    candidate_lncrnas: tuple[str, ...] = ()
    retrieved_papers: tuple[PubMedRecord, ...] = ()
    literature_evidence: tuple[EvidenceRecord, ...] = ()
    data_capabilities: DataCapabilities | None = None
    de_results: tuple[DifferentialExpressionResult, ...] = ()
    survival_results: tuple[SurvivalAnalysisResult, ...] = ()
    verification_report: VerificationReport | None = None
    observations: tuple[Observation, ...] = ()
    decisions: tuple[Decision, ...] = ()
    replanning_history: tuple[Decision, ...] = ()
    completed_steps: tuple[str, ...] = ()
    failed_steps: tuple[str, ...] = ()
    skipped_steps: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    proposed_alternatives: tuple[str, ...] = ()
    provenance: tuple[Provenance, ...] = ()
    status: ResearchStatus = ResearchStatus.INITIALIZED
    final_report: str | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> ResearchState:
        if self.tcga_project_id and not self.canonical_cancer_name:
            raise ValueError("TCGA project ID requires a canonical cancer name")

        step_sets = {
            "completed": set(self.completed_steps),
            "failed": set(self.failed_steps),
            "skipped": set(self.skipped_steps),
        }
        categories = tuple(step_sets)
        for index, left in enumerate(categories):
            for right in categories[index + 1 :]:
                overlap = step_sets[left] & step_sets[right]
                if overlap:
                    raise ValueError(
                        f"steps cannot be both {left} and {right}: {sorted(overlap)}"
                    )

        if self.status == ResearchStatus.COMPLETED and not self.final_report:
            raise ValueError("completed state requires a final report")
        return self

    def updated(self, **changes: Any) -> ResearchState:
        """Return a fully revalidated state with selected fields changed."""

        values = self.model_dump()
        values.update(changes)
        return ResearchState.model_validate(values)


class CheckpointMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    created_at: datetime
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    completed_step: str | None = None
    next_expected_step: str | None = None


class ResearchCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: CheckpointMetadata
    state: ResearchState

    @model_validator(mode="after")
    def validate_run_id(self) -> ResearchCheckpoint:
        if self.metadata.run_id != self.state.run_id:
            raise ValueError("checkpoint metadata and state run IDs must match")
        return self


def checkpoint_filename(sequence: int, slug: str) -> str:
    """Create a sortable, validated checkpoint filename."""

    if not 1 <= sequence <= 99:
        raise ValueError("checkpoint sequence must be between 1 and 99")
    if not _CHECKPOINT_SLUG.fullmatch(slug):
        raise ValueError("checkpoint slug must use lowercase words separated by underscores")
    return f"{sequence:02d}_{slug}.json"


def save_checkpoint(
    directory: Path,
    state: ResearchState,
    *,
    sequence: int,
    slug: str,
    completed_step: str | None = None,
    next_expected_step: str | None = None,
) -> Path:
    """Atomically save a new checkpoint without overwriting an existing one."""

    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / checkpoint_filename(sequence, slug)
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")

    checkpoint = ResearchCheckpoint(
        metadata=CheckpointMetadata(
            run_id=state.run_id,
            created_at=datetime.now(timezone.utc),
            completed_step=completed_step,
            next_expected_step=next_expected_step,
        ),
        state=state,
    )
    payload = checkpoint.model_dump_json(indent=2)

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_checkpoint(path: Path) -> ResearchCheckpoint:
    """Load and validate a checkpoint from disk."""

    with path.open(encoding="utf-8") as checkpoint_file:
        payload = json.load(checkpoint_file)
    return ResearchCheckpoint.model_validate(payload)


def resume_checkpoint(path: Path) -> ResearchState:
    """Restore state and continue at the checkpoint's next expected step."""

    checkpoint = load_checkpoint(path)
    next_step = checkpoint.metadata.next_expected_step
    if next_step is None:
        return checkpoint.state
    return checkpoint.state.updated(
        current_step=next_step,
        next_action=next_step,
        status=ResearchStatus.RUNNING,
    )
