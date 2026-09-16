from datetime import datetime, timezone
from types import SimpleNamespace as NS

from oncolncai.discovery_analysis import GenomeWideAnalysisRequest
from oncolncai.real_analysis import RealAnalysisRequest
from oncolncai.tcga_discovery import CapabilityStatus, ResearchMode
from oncolncai.ui import build_genome_wide_run_view


class Manifest(NS):
    def model_dump_json(self, indent=2):
        return '{"run_id":"ui-test"}'


def fixture(*, geo_status=CapabilityStatus.NOT_AVAILABLE):
    config = NS(normalization="median-ratio", method="PYDESEQ2", test_method="Wald",
                input_data_type="RAW_INTEGER_COUNTS", expression_transform="none",
                multiple_testing_method="Benjamini-Hochberg", alpha=.05)
    rows = (
        NS(gene_symbol="LINC1", feature_id="ENSG00000000001", case_mean_cpm=8.0,
           control_mean_cpm=2.0, log2_fold_change=2.0, p_value=.001,
           adjusted_p_value=.01, significant=True),
        NS(gene_symbol="LINC2", feature_id="ENSG00000000002", case_mean_cpm=3.0,
           control_mean_cpm=3.1, log2_fold_change=-.05, p_value=.8,
           adjusted_p_value=.9, significant=False),
    )
    de = NS(case_sample_count=50, control_sample_count=50, selected_feature_count=2, tested_feature_count=2,
            configuration=config, results=rows)
    candidate = NS(gene_symbol="LINC1", ensembl_gene_id="ENSG00000000001",
                   gene_biotype="lncRNA")
    discovery = NS(eligible_result_count=1, candidates=(candidate,), qualified_feature_ids=("ENSG00000000001",))
    survival = (
        NS(gene_symbol="LINC1", sample_count=100, event_count=30, hazard_ratio=1.8,
           confidence_interval_lower=1.2, confidence_interval_upper=2.7,
           p_value=.004, fdr=.008, ph_violation=False),
        NS(gene_symbol="LINC2", sample_count=98, event_count=29, hazard_ratio=.8,
           confidence_interval_lower=.5, confidence_interval_upper=1.3,
           p_value=.3, fdr=.3, ph_violation=None),
    )
    diagnostic = NS(term="expression_z", statistic=.2, p_value=.7, violation=False)
    adjusted = (NS(gene="LINC1", adjusted_hazard_ratio=1.7,
                   confidence_interval_lower=1.1, confidence_interval_upper=2.6,
                   p_value=.01, fdr=.02, initial_survival_eligible_count=100,
                   complete_case_count=90, event_count=28, missingness_by_covariate={"age": 10},
                   covariates_used=("expression_z", "age_z"),
                   convergence_status=NS(value="CONVERGED"), overall_ph_violation=False,
                   diagnostics=(diagnostic,)),)
    paper = NS(pmid="12345", title="LINC1 in LUAD", publication_year=2025,
               query="LINC1 AND LUAD", retrieval_timestamp=datetime(2026, 9, 15, tzinfo=timezone.utc))
    geo = (NS(accession="GSE1", platform="GPL1", tumor_count=12, normal_count=12,
              mapped_candidates=("LINC1",), unmapped_candidates=()),) if geo_status == CapabilityStatus.COMPLETED else ()
    rank = NS(gene_symbol="LINC1", ensembl_gene_id="ENSG00000000001", tier="Tier B",
              log2_fold_change=2.0, de_fdr=.01, survival_fdr=.008,
              adjusted_cox_fdr=.02, literature_pmids=("12345",),
              geo_expression_replication="NOT_AVAILABLE")
    finding = NS(check="ph_diagnostic:LINC1", decision=NS(value="ACCEPT"), detail="PH passed")
    event = NS(goal="Differential expression", execution_status=CapabilityStatus.COMPLETED,
               planned_action="Test counts", selected_tool="run_differential_expression",
               validated_arguments="TCGA-LUAD", observation="tested=2", decision="select candidates",
               replan_reason=None)
    prep = NS(source_columns=110, selected_samples=100, tumor_samples=50, normal_samples=50,
              duplicate_aliquots_removed=5, duplicate_gene_rows_removed=2,
              genes_in_source=60000, annotated_genes=59000, lncrnas_retained=2)
    provenance = (NS(source="NCI GDC", source_id="TCGA-LUAD", query="STAR counts", dataset_version="v1"),)
    manifest = Manifest(
        timestamp=datetime(2026, 9, 15, tzinfo=timezone.utc), upstream_source="NCI GDC",
        research_question="Which lncRNAs are promising biomarkers for lung adenocarcinoma?",
        distribution_source="GDC API", gene_annotation_source_version="GENCODE v36",
        expression_preprocessing="raw integer counts", expression_data_type="RAW_INTEGER_COUNTS",
        de_expression_source={"representation": "raw integer counts"},
        candidate_selection_rules={"fdr_threshold": .05, "absolute_log2_fc_threshold": 1,
                                   "minimum_mean_cpm": 1, "top_n": 10},
        geo_datasets_evaluated=("GSE1",), provenance=provenance,
        sample_counts={"de_raw_tumor": 50, "de_raw_normal": 50},
        output_artifacts=("de.csv", "manifest.json"),
    )
    return NS(
        status="COMPLETED_WITH_LIMITATIONS", run_id="ui-test",
        research_mode=ResearchMode.GENOME_WIDE_DISCOVERY,
        canonical_cancer_name="Lung Adenocarcinoma", project_id="TCGA-LUAD",
        manifest=manifest, preparation=NS(preparation=prep, clinical=tuple(range(95))),
        differential_expression=de, discovery=discovery, survival_screen=survival,
        adjusted_cox=adjusted, literature={"LINC1": (paper,)},
        literature_summary=NS(records_retrieved=1, unique_records=1, candidate_linked_records=1,
                              evidence_qualified_records=None, candidates_searched=1,
                              candidates_with_records=1), geo_replications=geo,
        rankings=(rank,), verification=(finding,), events=(event,),
        capabilities=(NS(analysis="geo_expression_replication", status=geo_status,
                         reason="No compatible dataset" if geo_status == CapabilityStatus.NOT_AVAILABLE else None),),
        limitations=("No compatible GEO survival cohort found.",), final_report="# Grounded report",
    )


def test_modes_validate_without_genome_wide_gene_and_with_single_candidate() -> None:
    GenomeWideAnalysisRequest(run_id="x", question="discover", cancer_name="lung adenocarcinoma")
    RealAnalysisRequest(run_id="x", question="validate", cancer_name="lung adenocarcinoma",
                        candidate="NEAT1", ensembl_gene_id="ENSG00000245532")


def test_full_real_projection_consumes_backend_results() -> None:
    view = build_genome_wide_run_view(fixture())
    assert "50" in view.overview and "synthetic" not in view.overview.casefold()
    assert view.de_rows[0][0] == "LINC1" and view.de_rows[0][5] == "2"
    assert "<svg" in view.volcano_html
    assert len(view.survival_rows) == 2
    assert view.adjusted_rows[0][1] == "1.7"
    assert view.ph_rows[0][-1] == "PASS"
    assert view.verifier_rows[0][1] == "ACCEPT"
    assert view.status == "COMPLETED WITH LIMITATIONS"
    assert "Genome-wide candidate discovery is not implemented" not in view.limitations


def test_geo_unavailable_and_integration_incomplete_remain_distinct() -> None:
    unavailable = build_genome_wide_run_view(fixture(geo_status=CapabilityStatus.NOT_AVAILABLE))
    incomplete = build_genome_wide_run_view(fixture(geo_status=CapabilityStatus.INTEGRATION_INCOMPLETE))
    assert "NOT AVAILABLE" in unavailable.geo_summary
    assert "INTEGRATION INCOMPLETE" in incomplete.geo_summary


def test_replanned_step_is_not_left_pending() -> None:
    result = fixture()
    skipped = result.events[0]
    skipped.execution_status = CapabilityStatus.SKIPPED_AFTER_REPLAN
    skipped.replan_reason = "insufficient normals"
    view = build_genome_wide_run_view(result)
    assert view.workflow_rows[0][1] == "SKIPPED AFTER REPLAN"
    assert view.workflow_rows[0][7] == "insufficient normals"
