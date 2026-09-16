from oncolncai.cancer_resolver import cancer_from_display_name, list_supported_cancers
from oncolncai.tcga_discovery import xena_expression_metadata_url
from oncolncai.tcga_survival import gdc_expression_url
from oncolncai.ui import build_genome_wide_run_view
from oncolncai.workflow_stages import WORKFLOW_STAGES, workflow_progress
from tests.test_ui_prompt2 import fixture


def test_canonical_registry_exposes_all_tcga_projects_and_source_configuration() -> None:
    entries = list_supported_cancers()
    assert len(entries) == 33
    assert len({item.tcga_project_id for item in entries}) == 33
    for project in ("TCGA-LUAD", "TCGA-BRCA", "TCGA-KIRC"):
        item = cancer_from_display_name(project)
        assert item.execution_status.value == "AVAILABLE"
        assert item.expression_source_template.format(project_id=project).endswith(f"{project}.star_counts.tsv.gz")
        assert item.raw_count_source and item.clinical_source and item.annotation_source


def test_project_drives_expression_urls_and_cache_namespace() -> None:
    cache_paths = set()
    for project in ("TCGA-LUAD", "TCGA-BRCA", "TCGA-KIRC"):
        assert project in xena_expression_metadata_url(project)
        assert project in gdc_expression_url(project)
        cache_paths.add(f".cache/oncolncai/gdc_raw_counts/{project}")
    assert len(cache_paths) == 3


def test_brca_final_projection_cannot_display_luad_identity() -> None:
    result = fixture()
    result.canonical_cancer_name = "Breast Invasive Carcinoma"
    result.project_id = "TCGA-BRCA"
    result.manifest.research_question = "Which lncRNAs are promising in breast invasive carcinoma?"
    result.manifest.provenance = tuple()
    result.literature = {}
    result.literature_summary = None
    report = build_genome_wide_run_view(result).report
    assert "TCGA-BRCA" in report and "Breast Invasive Carcinoma" in report
    assert "TCGA-LUAD" not in report and "lung adenocarcinoma" not in report.casefold()


def test_progress_uses_one_stable_denominator_and_stage_semantics() -> None:
    assert len(WORKFLOW_STAGES) == 16
    de = workflow_progress("differential_expression")
    assert (de.current_stage_index, de.total_stages, de.completed_stages, de.resolved_stages) == (7, 16, 6, 6)
    assert de.message == "Stage 7/16 — Differential expression"
