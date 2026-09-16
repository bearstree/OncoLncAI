"""Canonical user-visible stages for the real genome-wide workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkflowStage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stage_id: str
    label: str


class WorkflowProgress(BaseModel):
    """One atomic progress snapshot; resolved includes completed and terminal skips."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    current_stage_id: str
    current_stage_label: str
    current_stage_index: int = Field(ge=1)
    total_stages: int = Field(ge=1)
    completed_stages: int = Field(ge=0)
    skipped_or_unavailable_stages: int = Field(default=0, ge=0)
    detail: str | None = None

    @property
    def resolved_stages(self) -> int:
        return self.completed_stages + self.skipped_or_unavailable_stages

    @property
    def fraction(self) -> float:
        return min(0.99, max(0.0, (self.current_stage_index - 1) / self.total_stages))

    @property
    def message(self) -> str:
        suffix = f" — {self.detail}" if self.detail else ""
        return f"Stage {self.current_stage_index}/{self.total_stages} — {self.current_stage_label}{suffix}"


class StageCheckpoint(BaseModel):
    """Persisted stage state independent of browser rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    stage_id: str
    status: str
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: tuple[str, ...] = ()


WORKFLOW_STAGES: tuple[WorkflowStage, ...] = tuple(
    WorkflowStage(stage_id=stage_id, label=label)
    for stage_id, label in (
        ("cancer_resolution", "Cancer resolution"),
        ("capability_inspection", "Capability inspection"),
        ("expression_acquisition", "Expression acquisition"),
        ("clinical_acquisition", "Clinical acquisition"),
        ("data_qc", "Data QC"),
        ("lncrna_annotation", "lncRNA annotation"),
        ("differential_expression", "Differential expression"),
        ("candidate_discovery", "Candidate discovery"),
        ("survival_analysis", "Survival analysis"),
        ("adjusted_cox", "Adjusted Cox"),
        ("ph_diagnostics", "PH diagnostics"),
        ("literature_evidence", "Literature evidence"),
        ("geo_validation", "GEO validation"),
        ("evidence_ranking", "Evidence ranking"),
        ("verification", "Verification"),
        ("final_synthesis", "Final synthesis"),
    )
)


def workflow_progress(stage_id: str, *, detail: str | None = None,
                      skipped_or_unavailable: int = 0) -> WorkflowProgress:
    """Build a consistent snapshot from the sole workflow-stage registry."""

    index = next((i for i, stage in enumerate(WORKFLOW_STAGES, 1) if stage.stage_id == stage_id), None)
    if index is None:
        raise ValueError(f"Unknown workflow stage: {stage_id}")
    stage = WORKFLOW_STAGES[index - 1]
    return WorkflowProgress(
        current_stage_id=stage.stage_id, current_stage_label=stage.label,
        current_stage_index=index, total_stages=len(WORKFLOW_STAGES),
        completed_stages=max(0, index - 1 - skipped_or_unavailable),
        skipped_or_unavailable_stages=skipped_or_unavailable, detail=detail,
    )
