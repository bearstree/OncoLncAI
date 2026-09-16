import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from oncolncai import (
    Observation,
    Provenance,
    ResearchState,
    ResearchStatus,
    checkpoint_filename,
    load_checkpoint,
    resume_checkpoint,
    save_checkpoint,
)


def initialized_state() -> ResearchState:
    return ResearchState(
        run_id="run-001",
        research_question="Which lncRNAs are promising biomarkers for lung adenocarcinoma?",
    )


def test_state_can_be_created_and_revalidated_on_update() -> None:
    state = initialized_state()
    resolved = state.updated(
        user_cancer_name="lung adenocarcinoma",
        canonical_cancer_name="Lung Adenocarcinoma",
        cancer_synonyms=("LUAD",),
        tcga_project_id="TCGA-LUAD",
        completed_steps=("resolve_cancer",),
        next_action="retrieve_literature",
        status=ResearchStatus.RUNNING,
    )

    assert state.status == ResearchStatus.INITIALIZED
    assert resolved.tcga_project_id == "TCGA-LUAD"
    assert resolved.completed_steps == ("resolve_cancer",)

    with pytest.raises(ValidationError, match="canonical cancer name"):
        state.updated(tcga_project_id="TCGA-LUAD")


def test_step_cannot_have_conflicting_outcomes() -> None:
    with pytest.raises(ValidationError, match="cannot be both completed and skipped"):
        initialized_state().updated(
            completed_steps=("differential_expression",),
            skipped_steps=("differential_expression",),
        )


def test_completed_state_requires_report() -> None:
    with pytest.raises(ValidationError, match="requires a final report"):
        initialized_state().updated(status=ResearchStatus.COMPLETED)


def test_checkpoint_round_trip_and_resume() -> None:
    resolved = initialized_state().updated(
        canonical_cancer_name="Lung Adenocarcinoma",
        tcga_project_id="TCGA-LUAD",
        completed_steps=("resolve_cancer",),
        observations=(
            Observation(
                action="resolve_cancer",
                summary="Resolved to TCGA-LUAD using a curated mapping.",
                provenance=(Provenance(source="curated_mapping", source_id="TCGA-LUAD"),),
            ),
        ),
        next_action="retrieve_literature",
        status=ResearchStatus.PARTIAL,
    )

    with TemporaryDirectory(dir=".") as directory:
        path = save_checkpoint(
            Path(directory),
            resolved,
            sequence=2,
            slug="cancer_resolved",
            completed_step="resolve_cancer",
            next_expected_step="retrieve_literature",
        )
        loaded = load_checkpoint(path)
        resumed = resume_checkpoint(path)

        assert path.name == "02_cancer_resolved.json"
        assert loaded.state == resolved
        assert loaded.metadata.schema_version == "1.0"
        assert loaded.metadata.created_at.tzinfo is not None
        assert resumed.completed_steps == ("resolve_cancer",)
        assert resumed.current_step == "retrieve_literature"
        assert resumed.status == ResearchStatus.RUNNING


def test_checkpoint_does_not_persist_hidden_reasoning() -> None:
    with TemporaryDirectory(dir=".") as directory:
        path = save_checkpoint(
            Path(directory),
            initialized_state(),
            sequence=1,
            slug="initialized",
            next_expected_step="resolve_cancer",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))

    assert "chain_of_thought" not in json.dumps(payload)
    assert "reasoning" not in payload["state"]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResearchState.model_validate(
            {
                **payload["state"],
                "chain_of_thought": "private reasoning must not be persisted",
            }
        )


@pytest.mark.parametrize(
    ("sequence", "slug"),
    [(0, "initialized"), (100, "initialized"), (1, "Cancer Resolved"), (1, "../escape")],
)
def test_checkpoint_filename_rejects_invalid_values(sequence: int, slug: str) -> None:
    with pytest.raises(ValueError):
        checkpoint_filename(sequence, slug)


def test_save_does_not_overwrite_checkpoint() -> None:
    with TemporaryDirectory(dir=".") as directory:
        state = initialized_state()
        checkpoint_dir = Path(directory)
        save_checkpoint(checkpoint_dir, state, sequence=1, slug="initialized")

        with pytest.raises(FileExistsError, match="already exists"):
            save_checkpoint(checkpoint_dir, state, sequence=1, slug="initialized")


def test_load_rejects_unsupported_schema_version() -> None:
    with TemporaryDirectory(dir=".") as directory:
        path = save_checkpoint(
            Path(directory), initialized_state(), sequence=1, slug="initialized"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metadata"]["schema_version"] = "2.0"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValidationError, match="Input should be '1.0'"):
            load_checkpoint(path)
