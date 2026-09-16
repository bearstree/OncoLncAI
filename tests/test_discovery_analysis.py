from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from oncolncai.discovery_analysis import GenomeWideAnalysisRequest, GenomeWideAnalysisRunner
from oncolncai.schemas import Provenance, ResultStatus, ToolResult
from oncolncai.tcga_capabilities import DataCapabilities
from oncolncai.tcga_discovery import CapabilityStatus


class IneligibleRealCapability:
    def inspect(self, request):
        return ToolResult(
            status=ResultStatus.SUCCESS,
            data=DataCapabilities(
                project_id=request.project_id, rna_seq_available=True, clinical_available=True,
                survival_available=True, tumor_samples=30, normal_samples=2, survival_samples=25,
                de_recommended=False, survival_recommended=True, discovery_complete=True,
                reasons=("normal sample count below threshold",),
                checked_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
            ),
            provenance=[Provenance(source="real GDC fixture", source_id=request.project_id)],
        )


class MustNotRun:
    def prepare(self, request):
        raise AssertionError("ineligible DE must replan before expression retrieval")


def test_ineligible_real_de_replans_and_never_uses_synthetic_data() -> None:
    with TemporaryDirectory(dir=".") as directory:
        checkpoints = []
        result = GenomeWideAnalysisRunner(
            capability_client=IneligibleRealCapability(), genome_client=MustNotRun(),
            output_directory=Path(directory),
        ).run(GenomeWideAnalysisRequest(
            run_id="real-ineligible", cancer_name="lung adenocarcinoma",
            question="Which lncRNAs are promising biomarkers?",
        ), checkpoint=checkpoints.append)
        run_dir = Path(directory) / "real-ineligible"
        assert (run_dir / "stages" / "cancer_resolution.json").is_file()
        assert (run_dir / "stages" / "capability_inspection.json").is_file()
        assert (run_dir / "stages" / "differential_expression.json").is_file()
        assert (run_dir / "complete_result.json").is_file()
        run_state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
        assert run_state["status"] == "COMPLETED_WITH_LIMITATIONS"
        assert checkpoints[-1].stage_id == "final_synthesis"
    assert result.analysis_mode == "REAL"
    assert result.status == "COMPLETED_WITH_LIMITATIONS"
    assert result.capabilities[0].status == CapabilityStatus.NOT_ELIGIBLE
    skipped = [event for event in result.events if event.execution_status == CapabilityStatus.SKIPPED_AFTER_REPLAN]
    assert skipped and skipped[0].selected_tool == "run_differential_expression"
    serialized = result.model_dump_json().casefold()
    assert "synthetic" not in serialized
