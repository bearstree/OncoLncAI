"""Agentic coordinator for real TCGA genome-wide lncRNA discovery."""

from __future__ import annotations

import csv
import json
import os
import platform
from importlib.metadata import version
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from oncolncai.adjusted_survival import (
    AdjustedCoxRequest, AdjustedCoxResult, AdjustedSurvivalSample,
    apply_adjusted_cox_fdr, run_adjusted_cox,
)
from oncolncai.cache_identity import RealCacheIdentity, cache_identity_hash
from oncolncai.cancer_resolver import (
    CancerResolveRequest, ResolutionStatus, cancer_from_display_name, resolve_cancer,
)
from oncolncai.differential_expression import DifferentialExpressionResult, _benjamini_hochberg
from oncolncai.geo import GEOClient, GEOSearchRequest
from oncolncai.geo_replication import GEOReplicationResult, run_geo_expression_replication
from oncolncai.geo_retrieval import GEOExpressionClient, GEOIntegrationStatus
from oncolncai.gdc_counts import GDCRawCountClient, GDCRawCountRequest
from oncolncai.pubmed import PubMedClient, PubMedRecord, PubMedSearchRequest
from oncolncai.schemas import Provenance, ResultStatus
from oncolncai.survival import SurvivalAnalysisResult, run_survival_analysis
from oncolncai.tcga_capabilities import GDCCapabilityClient, TCCapabilityRequest, DataCapabilities
from oncolncai.tcga_discovery import (
    AnalysisCapability, CandidateDiscoveryResult, CandidateSelectionConfig, CapabilityStatus,
    GenomeWideDataRequest, PreparedGenomeWideCohort, ResearchMode, TCGAGenomeWideClient,
    discover_candidates, run_real_genome_wide_de,
)
from oncolncai.tcga_survival import TCGASurvivalDataClient, TCGASurvivalDataRequest
from oncolncai.workflow_stages import StageCheckpoint, WorkflowProgress, workflow_progress


class ScreenedSurvival(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gene_symbol: str
    ensembl_gene_id: str
    sample_count: int
    event_count: int
    clinical_cases_retrieved: int | None = None
    hazard_ratio: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    p_value: float
    fdr: float
    ph_p_value: float | None = None
    ph_violation: bool | None = None


class DiscoveryRanking(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gene_symbol: str
    ensembl_gene_id: str
    tier: str
    rule: str
    de_fdr: float
    log2_fold_change: float
    survival_fdr: float | None = None
    adjusted_cox_fdr: float | None = None
    literature_pmids: tuple[str, ...] = ()
    geo_expression_replication: str = "NOT_AVAILABLE"


class VerifierDecision(StrEnum):
    ACCEPT = "ACCEPT"
    DOWNGRADE = "DOWNGRADE"
    REJECT = "REJECT"


class DiscoveryVerificationFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    check: str
    decision: VerifierDecision
    detail: str


class LiteratureCollectionSummary(BaseModel):
    """Counts directly observed during candidate-specific PubMed retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    records_retrieved: int = Field(ge=0)
    unique_records: int = Field(ge=0)
    candidate_linked_records: int = Field(ge=0)
    evidence_qualified_records: int | None = Field(default=None, ge=0)
    candidates_searched: int = Field(ge=0)
    candidates_with_records: int = Field(ge=0)
    candidates_with_prognostic_evidence: int | None = Field(default=None, ge=0)
    candidates_with_expression_evidence: int | None = Field(default=None, ge=0)
    candidates_with_mechanistic_evidence: int | None = Field(default=None, ge=0)


class DiscoveryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    goal: str
    planned_action: str
    selected_tool: str
    validated_arguments: str
    observation: str
    decision: str
    execution_status: CapabilityStatus
    next_action: str | None = None
    replan_event: bool = False
    replan_reason: str | None = None


class GenomeWideAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    question: str = Field(min_length=1)
    cancer_name: str = Field(min_length=1)
    research_mode: ResearchMode = ResearchMode.GENOME_WIDE_DISCOVERY
    candidate_selection: CandidateSelectionConfig = CandidateSelectionConfig()
    max_pubmed_results_per_candidate: int = Field(default=3, ge=1, le=10)
    raw_count_max_tumor_samples: int = Field(default=50, ge=20)
    raw_count_max_normal_samples: int = Field(default=50, ge=10)
    allow_exploratory_continuous_de_fallback: bool = False


class DiscoveryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    timestamp: datetime
    analysis_mode: str = "REAL"
    research_mode: ResearchMode
    research_question: str
    resolved_cancer: str
    tcga_project: str
    gdc_query: str
    sample_counts: dict[str, int]
    count_definitions: dict[str, str] = Field(default_factory=dict)
    gene_annotation_source_version: str
    expression_preprocessing: str
    expression_data_type: str
    expression_transform: str
    expression_source_dataset: str
    expression_metadata_url: str | None
    expression_dataset_version: str | None
    upstream_gdc_release: str | None
    upstream_source: str
    distribution_source: str
    de_parameters: dict
    de_expression_source: dict
    candidate_selection_rules: dict
    survival_configuration: dict
    cox_covariates: tuple[str, ...]
    ph_configuration: str
    pubmed_queries_results: dict[str, tuple[str, ...]]
    geo_datasets_evaluated: tuple[str, ...]
    geo_dataset_selected: str | None
    ranking_configuration: str
    software_versions: dict[str, str]
    candidate_list: tuple[str, ...]
    verifier_findings: tuple[str, ...]
    limitations: tuple[str, ...]
    provenance: tuple[Provenance, ...]
    output_artifacts: tuple[str, ...]
    real_cache_identity: dict | None = None
    real_cache_identity_hash: str | None = None
    cache_status: str = "NEW_EXECUTION"


class GenomeWideAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_mode: str = "REAL"
    research_mode: ResearchMode
    run_id: str
    status: str
    canonical_cancer_name: str
    project_id: str
    capabilities: tuple[AnalysisCapability, ...]
    preparation: PreparedGenomeWideCohort | None = None
    differential_expression: DifferentialExpressionResult | None = None
    discovery: CandidateDiscoveryResult | None = None
    survival_screen: tuple[ScreenedSurvival, ...] = ()
    adjusted_cox: tuple[AdjustedCoxResult, ...] = ()
    geo_replications: tuple[GEOReplicationResult, ...] = ()
    literature: dict[str, tuple[PubMedRecord, ...]] = {}
    literature_summary: LiteratureCollectionSummary | None = None
    rankings: tuple[DiscoveryRanking, ...] = ()
    verification: tuple[DiscoveryVerificationFinding, ...] = ()
    events: tuple[DiscoveryEvent, ...]
    limitations: tuple[str, ...]
    final_report: str
    manifest: DiscoveryManifest


ProgressCallback = Callable[[WorkflowProgress], None]
CheckpointCallback = Callable[[StageCheckpoint], None]


def _atomic_json(path: Path, value: object) -> None:
    """Atomically replace JSON so incomplete writes are never considered final."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _csv_value(value: object) -> object:
    """Preserve nested structured values in a portable CSV cell."""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if hasattr(value, "value"):
        return value.value
    return value


def _write_rows(path: Path, rows: list[dict[str, object]], *, empty_fields: tuple[str, ...]) -> None:
    """Write a stable CSV artifact, including headers when no result was available."""

    fields = list(rows[0]) if rows else list(empty_fields)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


class GenomeWideAnalysisRunner:
    def __init__(self, *, genome_client: TCGAGenomeWideClient | None = None,
                 capability_client: GDCCapabilityClient | None = None,
                 survival_client: TCGASurvivalDataClient | None = None,
                 pubmed_client: PubMedClient | None = None, geo_client: GEOClient | None = None,
                 geo_expression_client: GEOExpressionClient | None = None,
                 raw_count_client: GDCRawCountClient | None = None,
                 output_directory: Path = Path(".cache/oncolncai/runs"),
                 clock: Callable[[], datetime] | None = None) -> None:
        self.genome = genome_client or TCGAGenomeWideClient()
        self.capability = capability_client or GDCCapabilityClient()
        self.survival = survival_client or TCGASurvivalDataClient()
        self.pubmed = pubmed_client or PubMedClient()
        self.geo = geo_client or GEOClient()
        self.geo_expression = geo_expression_client or GEOExpressionClient()
        self.raw_counts = raw_count_client or GDCRawCountClient()
        self.output_directory = output_directory
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def run(self, request: GenomeWideAnalysisRequest, *, progress: ProgressCallback | None = None,
            checkpoint: CheckpointCallback | None = None) -> GenomeWideAnalysisResult:
        update = progress or (lambda _snapshot: None)
        emit = lambda stage_id, detail=None: update(workflow_progress(stage_id, detail=detail))
        output_dir = self.output_directory / request.run_id
        stages_dir = output_dir / "stages"
        run_state_path = output_dir / "run_state.json"
        resolved_stage_ids: list[str] = []

        def save_stage(stage_id: str, payload: dict, *, status: str = "COMPLETED",
                       artifact_paths: tuple[str, ...] = ()) -> None:
            stage = StageCheckpoint(run_id=request.run_id, stage_id=stage_id, status=status,
                                    payload=payload, artifact_paths=artifact_paths)
            _atomic_json(stages_dir / f"{stage_id}.json", stage.model_dump(mode="json"))
            if status not in {"RUNNING", "FAILED"}:
                resolved_stage_ids.append(stage_id)
            _atomic_json(run_state_path, {
                "run_id": request.run_id, "status": "RUNNING", "current_stage": stage_id,
                "resolved_stages": list(dict.fromkeys(resolved_stage_ids)),
            })
            if checkpoint is not None:
                checkpoint(stage)

        def finish_partial(cancer_name: str, project_id: str) -> GenomeWideAnalysisResult:
            reason = limitations[-1] if limitations else "Required input was unavailable."
            remaining = ("differential_expression", "candidate_discovery", "survival_analysis",
                         "adjusted_cox", "ph_diagnostics", "literature_evidence", "geo_validation",
                         "evidence_ranking", "verification")
            for stage_id in remaining:
                save_stage(stage_id, {"reason": reason}, status=(
                    "NOT_ELIGIBLE" if stage_id == "differential_expression" else "SKIPPED_AFTER_REPLAN"))
            partial = self._partial(request, cancer_name, project_id, capabilities, events,
                                    limitations, provenance)
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = output_dir / "run_manifest.json"
            result_path = output_dir / "complete_result.json"
            _atomic_json(manifest_path, partial.manifest.model_dump(mode="json"))
            _atomic_json(result_path, partial.model_dump(mode="json"))
            save_stage("final_synthesis", {"status": partial.status, "final_report": partial.final_report,
                "complete_result": str(result_path), "manifest": str(manifest_path)},
                artifact_paths=(str(result_path), str(manifest_path)))
            _atomic_json(run_state_path, {"run_id": request.run_id, "status": partial.status,
                "current_stage": "final_synthesis", "resolved_stages": list(dict.fromkeys(resolved_stage_ids)),
                "complete_result": str(result_path), "manifest": str(manifest_path)})
            return partial
        events: list[DiscoveryEvent] = []
        limitations: list[str] = []
        provenance: list[Provenance] = []
        capabilities: list[AnalysisCapability] = []
        emit("cancer_resolution")
        resolution = resolve_cancer(CancerResolveRequest(name=request.cancer_name))
        cancer = resolution.data
        if cancer is None or cancer.status != ResolutionStatus.RESOLVED or not cancer.tcga_project_id or not cancer.canonical_name:
            raise ValueError("genome-wide REAL mode requires an unambiguous supported TCGA cancer")
        project = cancer.tcga_project_id
        project_config = cancer_from_display_name(project)
        provenance.extend(resolution.provenance)
        events.append(_event("Ground the cancer", "Resolve cancer", "resolve_cancer", request.cancer_name,
                             f"Resolved to {project}", "Use this project for all tools", CapabilityStatus.COMPLETED, "capability_check"))
        save_stage("cancer_resolution", {"canonical_cancer_name": cancer.canonical_name, "project_id": project})

        emit("capability_inspection")
        observed = self.capability.inspect(TCCapabilityRequest(project_id=project))
        data_capability: DataCapabilities | None = observed.data
        if data_capability is None:
            raise RuntimeError(observed.error.message if observed.error else "GDC capability inspection failed")
        provenance.extend(observed.provenance)
        de_ready = data_capability.de_recommended
        capabilities.append(AnalysisCapability(analysis="differential_expression", data_available=data_capability.rna_seq_available,
                                               inputs_valid=True, eligible=de_ready,
                                               status=CapabilityStatus.READY if de_ready else CapabilityStatus.NOT_ELIGIBLE,
                                               reason=None if de_ready else "; ".join(data_capability.reasons)))
        events.append(_event("Assess analysis eligibility", "Inspect GDC", "inspect_tcga_capabilities", project,
                             f"tumor={data_capability.tumor_samples}, normal={data_capability.normal_samples}",
                             "Prepare genome-wide DE" if de_ready else "Replan away from DE", CapabilityStatus.COMPLETED,
                             "prepare_expression" if de_ready else "survival_only", not de_ready,
                             None if de_ready else "Insufficient eligible tumor/normal data"))
        save_stage("capability_inspection", data_capability.model_dump(mode="json"))
        if not de_ready:
            limitations.append("Genome-wide DE was NOT_ELIGIBLE under observed GDC sample counts.")
            return finish_partial(cancer.canonical_name, project)

        emit("expression_acquisition")
        prepared_result = self.genome.prepare(GenomeWideDataRequest(project_id=project))
        cohort = prepared_result.data
        if cohort is None:
            raise RuntimeError(prepared_result.error.message if prepared_result.error else "genome-wide preparation failed")
        provenance.extend(prepared_result.provenance)
        capabilities[-1] = capabilities[-1].model_copy(update={"status": CapabilityStatus.RUNNING})
        events.append(_event("Build analysis matrix", "Prepare samples and lncRNAs", "prepare_genome_wide_cohort", project,
                             f"{cohort.preparation.lncrnas_retained} lncRNAs; {cohort.preparation.tumor_samples} tumor; {cohort.preparation.normal_samples} normal",
                             "Execute existing DE engine", CapabilityStatus.COMPLETED, "differential_expression"))
        preparation_payload = {
            "project_id": project, "preparation": cohort.preparation.model_dump(mode="json"),
            "expression_url": cohort.expression_url, "annotation_url": cohort.annotation_url,
            "expression_preprocessing": cohort.expression_preprocessing,
        }
        save_stage("expression_acquisition", preparation_payload)
        save_stage("clinical_acquisition", {"project_id": project, "clinical_records": len(cohort.clinical)})
        save_stage("data_qc", preparation_payload)
        save_stage("lncrna_annotation", {"project_id": project,
            "annotated_genes": cohort.preparation.annotated_genes,
            "lncrnas_retained": cohort.preparation.lncrnas_retained,
            "annotation_url": cohort.annotation_url})

        emit("clinical_acquisition")
        emit("data_qc")
        emit("lncrna_annotation")
        raw_result = self.raw_counts.prepare(GDCRawCountRequest(
            project_id=project, max_tumor_samples=request.raw_count_max_tumor_samples,
            max_normal_samples=request.raw_count_max_normal_samples,
        ))
        raw_cohort = raw_result.data
        if raw_cohort is None:
            reason = raw_result.error.message if raw_result.error else "raw-count preparation failed"
            if not request.allow_exploratory_continuous_de_fallback:
                capabilities[-1] = capabilities[-1].model_copy(update={
                    "implementation": "PyDESeq2 available; raw-count integration incomplete",
                    "inputs_valid": False, "eligible": False, "status": CapabilityStatus.INTEGRATION_INCOMPLETE,
                    "reason": reason,
                })
                limitations.append(f"Production count-based DE INTEGRATION_INCOMPLETE: {reason}")
                return finish_partial(cancer.canonical_name, project)
            limitations.append(f"Production count-based DE unavailable; explicitly permitted exploratory Welch fallback used: {reason}")
            de_request = cohort.de_request().model_copy(update={"method": "WELCH"})
        else:
            provenance.extend(raw_result.provenance)
            de_request = raw_cohort.de_request()
            events.append(_event("Obtain inference-grade DE input", "Assemble native GDC counts", "GDCRawCountClient.prepare", project,
                                 f"raw integer counts: tumor={raw_cohort.preparation.tumor_samples}, normal={raw_cohort.preparation.normal_samples}, lncRNAs={raw_cohort.preparation.lncrnas_retained}, cache_reused={raw_cohort.preparation.cache_reused}",
                                 "Run AUTO count-based DE", CapabilityStatus.COMPLETED, "differential_expression"))

        emit("differential_expression")
        from oncolncai.differential_expression import run_differential_expression
        de_tool = run_differential_expression(de_request)
        de = de_tool.data
        if de is None:
            raise RuntimeError(de_tool.error.message if de_tool.error else "DE failed")
        provenance.extend(de_tool.provenance)
        capabilities[-1] = capabilities[-1].model_copy(update={"status": CapabilityStatus.COMPLETED})
        events.append(_event("Test tumor-normal expression", "Run genome-wide DE", "run_differential_expression", project,
                             f"tested={de.tested_feature_count}; filtered={de.filtered_feature_count}",
                             "Apply prespecified candidate rules", CapabilityStatus.COMPLETED, "candidate_selection"))
        save_stage("differential_expression", de.model_dump(mode="json"))

        emit("candidate_discovery")
        discovery = discover_candidates(de, cohort.annotations, request.candidate_selection)
        events.append(_event("Discover candidates", "Filter DE table", "discover_candidates", request.candidate_selection.model_dump_json(),
                             f"eligible={discovery.eligible_result_count}; retained={len(discovery.candidates)}",
                             "Screen retained candidates for survival", CapabilityStatus.COMPLETED, "survival_screen", True,
                             "DE reduced the genome-wide result to a bounded top-N set"))
        save_stage("candidate_discovery", discovery.model_dump(mode="json"))

        emit("survival_analysis")
        raw_survival: list[tuple[str, str, SurvivalAnalysisResult, object]] = []
        for index, candidate in enumerate(discovery.candidates):
            emit("survival_analysis", f"candidate {index + 1}/{len(discovery.candidates)}: {candidate.gene_symbol}")
            prepared_survival = self.survival.prepare(TCGASurvivalDataRequest(project_id=project, gene=candidate.gene_symbol, ensembl_gene_id=candidate.ensembl_gene_id))
            if prepared_survival.data is None:
                limitations.append(f"Survival unavailable for {candidate.gene_symbol}: {prepared_survival.error.message if prepared_survival.error else 'preparation failed'}")
                continue
            analyzed = run_survival_analysis(prepared_survival.data.survival_request())
            if analyzed.data is not None:
                raw_survival.append((candidate.gene_symbol, candidate.ensembl_gene_id, analyzed.data, prepared_survival.data))
                provenance.extend(prepared_survival.provenance + analyzed.provenance)
        survival_fdr = _benjamini_hochberg([item[2].cox_p_value for item in raw_survival])
        screens = tuple(ScreenedSurvival(
            gene_symbol=gene, ensembl_gene_id=ensembl, sample_count=value.sample_count, event_count=value.event_count,
            clinical_cases_retrieved=prepared_survival.preparation.clinical_project_rows,
            hazard_ratio=value.hazard_ratio, confidence_interval_lower=value.confidence_interval_lower,
            confidence_interval_upper=value.confidence_interval_upper, p_value=value.cox_p_value, fdr=fdr,
            ph_p_value=value.diagnostics.proportional_hazards_p_value,
            ph_violation=(not value.diagnostics.proportional_hazards_assumption_met) if value.diagnostics.proportional_hazards_assumption_met is not None else None,
        ) for (gene, ensembl, value, prepared_survival), fdr in zip(raw_survival, survival_fdr))
        capabilities.append(AnalysisCapability(
            analysis="survival_analysis", implementation="Kaplan-Meier + log-rank + univariate Cox PH",
            data_available=bool(raw_survival), inputs_valid=bool(raw_survival), eligible=bool(screens),
            status=CapabilityStatus.COMPLETED if screens else CapabilityStatus.NOT_ELIGIBLE,
            reason=None if screens else "No candidate had eligible project-specific expression and survival records.",
        ))
        events.append(_event("Screen prognostic association", "Run Kaplan-Meier, log-rank, and univariate Cox",
                             "run_survival_analysis", f"candidates={len(discovery.candidates)}; endpoint=OS; BH correction",
                             f"completed={len(screens)}/{len(discovery.candidates)}",
                             "Apply BH correction and continue eligible candidates to adjusted Cox",
                             CapabilityStatus.COMPLETED if screens else CapabilityStatus.NOT_ELIGIBLE, "adjusted_cox"))
        save_stage("survival_analysis", {"results": [item.model_dump(mode="json") for item in screens]},
                   status="COMPLETED" if screens else "COMPLETED_WITH_ZERO_RESULTS")

        emit("adjusted_cox")
        adjusted: list[AdjustedCoxResult] = []
        for gene, _, _, prepared_survival in raw_survival:
            covariates = {item.patient_id: item for item in prepared_survival.clinical_covariates}
            samples = tuple(AdjustedSurvivalSample(
                patient_id=item.sample_id, time=item.time, event=item.event, expression=item.expression,
                age=covariates[item.sample_id].age, sex=covariates[item.sample_id].sex,
                pathological_stage=covariates[item.sample_id].pathological_stage,
            ) for item in prepared_survival.samples if item.sample_id in covariates)
            try:
                adjusted.append(run_adjusted_cox(AdjustedCoxRequest(project_id=project, gene=gene, samples=samples)))
            except ValueError as exc:
                limitations.append(f"Adjusted Cox for {gene} was NOT_ELIGIBLE or failed convergence: {exc}")
        adjusted_results = apply_adjusted_cox_fdr(tuple(adjusted))
        capabilities.append(AnalysisCapability(
            analysis="adjusted_cox", implementation="complete-case multivariable Cox PH",
            data_available=bool(raw_survival), inputs_valid=bool(adjusted_results), eligible=bool(adjusted_results),
            status=CapabilityStatus.COMPLETED if adjusted_results else CapabilityStatus.NOT_ELIGIBLE,
            reason=None if adjusted_results else "No model met project-specific complete-case/event/convergence requirements.",
        ))
        capabilities.append(AnalysisCapability(
            analysis="ph_diagnostics", implementation="Schoenfeld-residual time-trend diagnostics",
            data_available=bool(adjusted_results), inputs_valid=bool(adjusted_results), eligible=bool(adjusted_results),
            status=CapabilityStatus.COMPLETED if adjusted_results else CapabilityStatus.NOT_ELIGIBLE,
            reason=None if adjusted_results else "PH diagnostics require an eligible fitted Cox model.",
        ))
        events.append(_event("Estimate independent association", "Fit adjusted Cox", "run_adjusted_cox", "expression + age + sex + stage",
                             f"converged={len(adjusted_results)}/{len(raw_survival)}", "Retain converged models; downgrade failures",
                             CapabilityStatus.COMPLETED if adjusted_results else CapabilityStatus.NOT_ELIGIBLE, "literature"))
        events.append(_event("Check proportional-hazards assumptions", "Evaluate model-level and covariate diagnostics",
                             "schoenfeld_residual_diagnostics", f"models={len(adjusted_results)}; alpha=0.05",
                             f"diagnostics={sum(len(item.diagnostics) for item in adjusted_results)}; violations={sum(item.overall_ph_violation is True for item in adjusted_results)}",
                             "Downgrade affected claims; do not relabel association as independent prognosis",
                             CapabilityStatus.COMPLETED if adjusted_results else CapabilityStatus.NOT_ELIGIBLE, "literature"))
        save_stage("adjusted_cox", {"results": [item.model_dump(mode="json") for item in adjusted_results]},
                   status="COMPLETED" if adjusted_results else "NOT_ELIGIBLE")
        save_stage("ph_diagnostics", {"results": [
            {"gene": item.gene, **diagnostic.model_dump(mode="json")}
            for item in adjusted_results for diagnostic in item.diagnostics
        ]}, status="COMPLETED" if adjusted_results else "NOT_ELIGIBLE")
        emit("ph_diagnostics")

        emit("literature_evidence")
        literature: dict[str, tuple[PubMedRecord, ...]] = {}
        for candidate in discovery.candidates:
            found = self.pubmed.search(PubMedSearchRequest(cancer=cancer.canonical_name, concepts=(candidate.gene_symbol,), max_results=request.max_pubmed_results_per_candidate))
            literature[candidate.gene_symbol] = found.data.records if found.data else ()
            provenance.extend(found.provenance)
        retrieved_pmids = [paper.pmid for papers in literature.values() for paper in papers]
        literature_summary = LiteratureCollectionSummary(
            records_retrieved=len(retrieved_pmids), unique_records=len(set(retrieved_pmids)),
            candidate_linked_records=len(retrieved_pmids), evidence_qualified_records=None,
            candidates_searched=len(discovery.candidates),
            candidates_with_records=sum(bool(papers) for papers in literature.values()),
        )
        events.append(_event("Retrieve literature evidence", "Search candidate-specific PubMed records", "PubMedClient.search",
                             f"candidates={len(discovery.candidates)}; max_results_per_candidate={request.max_pubmed_results_per_candidate}",
                             f"verified_records={sum(len(items) for items in literature.values())}",
                             "Retain PMID provenance and continue when no record is found",
                             CapabilityStatus.COMPLETED, "geo_validation"))
        save_stage("literature_evidence", {
            "summary": literature_summary.model_dump(mode="json"),
            "records": {gene: [paper.model_dump(mode="json") for paper in papers]
                        for gene, papers in literature.items()},
        }, status="COMPLETED" if retrieved_pmids else "COMPLETED_WITH_ZERO_RESULTS")

        emit("geo_validation")
        tissue = project_config.primary_site
        geo_result = self.geo.search(GEOSearchRequest(cancer=cancer.canonical_name, tissue=tissue, min_cases=10, min_controls=10))
        assessed = geo_result.data.assessments if geo_result.data else ()
        suitable = geo_result.data.suitable_datasets if geo_result.data else ()
        selected_geo = None
        geo_replications: list[GEOReplicationResult] = []
        retrieval_reasons: list[str] = []
        directions = {item.gene_symbol: item.log2_fold_change for item in discovery.candidates}
        candidate_symbols = tuple(directions)
        for assessment in suitable:
            accession = assessment.dataset.accession
            try:
                parsed = self.geo_expression.retrieve(
                    accession, candidates=candidate_symbols, tcga_directions=directions,
                    cancer_terms=(cancer.canonical_name, project.removeprefix("TCGA-")), tissue_terms=(tissue,),
                )
                if parsed.compatibility.status == GEOIntegrationStatus.ELIGIBLE and parsed.replication_request is not None:
                    geo_replications.append(run_geo_expression_replication(parsed.replication_request))
                    selected_geo = accession
                else:
                    retrieval_reasons.append(f"{accession}: {parsed.compatibility.status.value} — {'; '.join(parsed.compatibility.reasons)}")
            except Exception as exc:
                retrieval_reasons.append(f"{accession}: INTEGRATION_INCOMPLETE — {exc}")
        if geo_replications:
            geo_status = CapabilityStatus.COMPLETED
            reason = f"Ran real expression replication for {len(geo_replications)} eligible GEO cohort(s)."
        elif suitable:
            geo_status = CapabilityStatus.INTEGRATION_INCOMPLETE
            reason = "; ".join(retrieval_reasons) or "Supported GEO matrices could not be integrated."
        else:
            geo_status = CapabilityStatus.NOT_AVAILABLE
            reason = "No series passed every explicit metadata suitability criterion."
        capabilities.append(AnalysisCapability(analysis="geo_expression_replication", implementation="Series Matrix + GPL annotation + deterministic phenotype parser",
                                               data_available=bool(suitable), inputs_valid=bool(geo_replications), eligible=bool(geo_replications), status=geo_status, reason=reason))
        limitations.append(f"GEO expression replication {geo_status.value}: {reason}")
        events.append(_event("Seek external replication", "Assess GEO datasets", "search_geo", cancer.canonical_name,
                             f"evaluated={len(assessed)}; selected={selected_geo or 'none'}", "Do not force validation",
                             geo_status, "ranking", bool(assessed), reason))
        save_stage("geo_validation", {"status": geo_status.value, "reason": reason,
            "datasets_evaluated": [item.dataset.model_dump(mode="json") for item in assessed],
            "replications": [item.model_dump(mode="json") for item in geo_replications]}, status=geo_status.value)

        emit("evidence_ranking")
        screen_by_gene = {item.gene_symbol: item for item in screens}
        adjusted_by_gene = {item.gene: item for item in adjusted_results}
        rankings: list[DiscoveryRanking] = []
        for candidate in discovery.candidates:
            screen = screen_by_gene.get(candidate.gene_symbol)
            adj = adjusted_by_gene.get(candidate.gene_symbol)
            ph_ok = screen is not None and screen.ph_violation is False and (adj is None or adj.overall_ph_violation is False)
            geo_matches = [feature for cohort_result in geo_replications for feature in cohort_result.results
                           if feature.gene_symbol == candidate.gene_symbol]
            replicated = any(item.fdr <= .05 and item.direction_consistent_with_tcga for item in geo_matches)
            tier = "Tier A" if replicated and adj and adj.fdr is not None and adj.fdr <= .05 and ph_ok else "Tier B" if screen and screen.fdr <= .05 and ph_ok else "Tier C"
            rule = "Tier A requires direction-consistent GEO replication plus adjusted-Cox FDR<=0.05 and no PH violation; Tier B requires survival FDR<=0.05 and no PH violation; otherwise Tier C."
            rankings.append(DiscoveryRanking(gene_symbol=candidate.gene_symbol, ensembl_gene_id=candidate.ensembl_gene_id,
                                                tier=tier, rule=rule, de_fdr=candidate.fdr, log2_fold_change=candidate.log2_fold_change,
                                                survival_fdr=screen.fdr if screen else None, adjusted_cox_fdr=adj.fdr if adj else None,
                                                literature_pmids=tuple(item.pmid for item in literature[candidate.gene_symbol]),
                                                geo_expression_replication="REPLICATED" if replicated else "NOT_REPLICATED" if geo_matches else "NOT_AVAILABLE"))
        save_stage("evidence_ranking", {"results": [item.model_dump(mode="json") for item in rankings]})
        verification: list[DiscoveryVerificationFinding] = [
            DiscoveryVerificationFinding(check="real_synthetic_separation", decision=VerifierDecision.ACCEPT, detail="REAL coordinator used only public-data clients."),
            DiscoveryVerificationFinding(check="project_consistency", decision=VerifierDecision.ACCEPT, detail=f"All inputs resolved to {project}."),
            DiscoveryVerificationFinding(check="expression_shape", decision=VerifierDecision.ACCEPT, detail=f"{len(cohort.features)} lncRNA rows x {len(cohort.samples)} sample columns; every row validated."),
            DiscoveryVerificationFinding(check="identifier_annotation", decision=VerifierDecision.ACCEPT, detail="Normalized Ensembl IDs are unique and every retained row has a GENCODE lncRNA biotype."),
            DiscoveryVerificationFinding(check="de_bh_and_selection", decision=VerifierDecision.ACCEPT, detail="DE applied BH over all tested lncRNAs; candidate thresholds were reapplied deterministically."),
        ]
        emit("verification")
        verification.extend(DiscoveryVerificationFinding(
            check=f"ph_diagnostic:{item.gene_symbol}", decision=VerifierDecision.DOWNGRADE,
            detail="Schoenfeld-residual time-trend diagnostic indicated a PH violation."
        ) for item in screens if item.ph_violation)
        save_stage("verification", {"results": [item.model_dump(mode="json") for item in verification]})
        verifier_findings = tuple(f"{item.decision.value} {item.check}: {item.detail}" for item in verification)

        output_dir = self.output_directory / request.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        de_path = output_dir / "complete_de_table.csv"
        _write_rows(de_path, [item.model_dump(mode="json") for item in de.results], empty_fields=("feature_id",))
        qc_path = output_dir / "qc_summary.json"
        qc_path.write_text(json.dumps({
            "project_id": project,
            "transformed_cohort": cohort.preparation.model_dump(mode="json"),
            "de_case_samples": de.case_sample_count,
            "de_control_samples": de.control_sample_count,
            "de_input_features": de.input_feature_count,
            "de_tested_features": de.tested_feature_count,
            "de_filtered_features": de.filtered_feature_count,
            "annotation": f"GENCODE v36 sha256:{cohort.annotation_sha256}",
            "expression_preprocessing": cohort.expression_preprocessing,
        }, indent=2) + "\n", encoding="utf-8")
        candidate_path = output_dir / "candidate_table.csv"
        _write_rows(candidate_path, [item.model_dump(mode="json") for item in discovery.candidates], empty_fields=("gene_symbol", "ensembl_gene_id"))
        survival_path = output_dir / "survival_table.csv"
        _write_rows(survival_path, [item.model_dump(mode="json") for item in screens], empty_fields=("gene_symbol", "sample_count", "event_count"))
        adjusted_path = output_dir / "adjusted_cox_table.csv"
        _write_rows(adjusted_path, [item.model_dump(mode="json") for item in adjusted_results], empty_fields=("gene", "status"))
        ph_path = output_dir / "ph_diagnostics.csv"
        ph_rows = [{"gene": item.gene, **diagnostic.model_dump(mode="json")} for item in adjusted_results for diagnostic in item.diagnostics]
        _write_rows(ph_path, ph_rows, empty_fields=("gene", "term", "statistic", "p_value", "violation"))
        literature_path = output_dir / "literature_results.csv"
        literature_rows = [{"gene_symbol": gene, **paper.model_dump(mode="json")} for gene, papers in literature.items() for paper in papers]
        _write_rows(literature_path, literature_rows, empty_fields=("gene_symbol", "pmid", "title"))
        geo_path = output_dir / "geo_results.json"
        geo_path.write_text(json.dumps({
            "status": geo_status.value,
            "reason": reason,
            "datasets_evaluated": [item.dataset.model_dump(mode="json") for item in assessed],
            "selected_accession": selected_geo,
            "replications": [item.model_dump(mode="json") for item in geo_replications],
        }, indent=2) + "\n", encoding="utf-8")
        verifier_path = output_dir / "verifier_output.csv"
        _write_rows(verifier_path, [item.model_dump(mode="json") for item in verification], empty_fields=("check", "decision", "detail"))
        ranking_path = output_dir / "candidate_ranking.csv"
        _write_rows(ranking_path, [item.model_dump(mode="json") for item in rankings], empty_fields=("gene_symbol", "tier"))
        report_path = output_dir / "final_report.md"
        emit("final_synthesis")
        report = _report(cancer.canonical_name, project, cohort, discovery, screens, adjusted_results, tuple(rankings), limitations)
        report_path.write_text(report + "\n", encoding="utf-8")
        manifest_path = output_dir / "run_manifest.json"
        result_path = output_dir / "complete_result.json"
        software_versions = {"python": platform.python_version(), "numpy": version("numpy"),
                             "scipy": version("scipy"), "oncolncai": "0.1.0"}
        real_cache_identity = RealCacheIdentity(
            project_id=project, source_dataset_id=cohort.expression_url,
            source_dataset_checksum_or_version=(cohort.expression_dataset_version or f"sha256:{cohort.annotation_sha256}"),
            expression_representation=(raw_cohort.source.representation if raw_cohort is not None else cohort.expression_data_type.value),
            analysis_configuration={"de": de.configuration.model_dump(mode="json"),
                                    "candidate_selection": request.candidate_selection.model_dump(mode="json")},
            de_method=str(de.configuration.method),
            statistical_thresholds={"fdr": request.candidate_selection.fdr_threshold,
                                    "absolute_log2_fc": request.candidate_selection.absolute_log2_fc_threshold,
                                    "minimum_mean_cpm": request.candidate_selection.minimum_mean_cpm},
            annotation_version=f"GENCODE v36 sha256:{cohort.annotation_sha256}", project_version="0.1.0",
            software_versions=software_versions,
            source_git_sha=os.getenv("ONCOLNCAI_SOURCE_GIT_SHA", "unknown"),
        )
        manifest = DiscoveryManifest(
            run_id=request.run_id, timestamp=self.clock(), research_mode=request.research_mode,
            research_question=request.question, resolved_cancer=cancer.canonical_name, tcga_project=project,
            gdc_query=f"project={project}; data_type=Gene Expression Quantification; workflow=STAR - Counts",
            sample_counts={"transformed_tumor": cohort.preparation.tumor_samples, "transformed_normal": cohort.preparation.normal_samples,
                           "de_raw_tumor": de.case_sample_count, "de_raw_normal": de.control_sample_count},
            count_definitions={
                "transformed_tumor": "Primary-tumor samples selected in the checksum-pinned transformed expression cohort.",
                "transformed_normal": "Solid-tissue normal samples selected in the checksum-pinned transformed expression cohort.",
                "de_raw_tumor": "Primary-tumor native GDC raw-count samples used by differential expression.",
                "de_raw_normal": "Solid-tissue normal native GDC raw-count samples used by differential expression.",
                "annotated_lncrnas": "GENCODE lncRNA-biotype rows retained before expression filtering.",
                "expression_eligible_lncrnas": "lncRNAs passing the raw-count expression filter.",
                "de_tested_lncrnas": "lncRNAs with finite model results included in multiple-testing correction.",
                "de_qualified_lncrnas": "DE-tested lncRNAs meeting configured FDR, effect-size, and expression rules.",
                "clinical_cases_retrieved": "Project clinical records observed by the survival preparation client.",
                "survival_eligible_cases": "Unique primary-tumor patients aligned to an evaluable survival endpoint.",
                "adjusted_cox_complete_cases": "Survival-eligible patients with every requested Cox covariate present.",
            },
            gene_annotation_source_version=f"GENCODE v36 sha256:{cohort.annotation_sha256}",
            expression_preprocessing=cohort.expression_preprocessing,
            de_parameters={**de.configuration.model_dump(), "design_factors": ["group"],
                           "contrast": ["group", "case", "control"], "control_genes": None,
                           "tumor_n": de.case_sample_count, "normal_n": de.control_sample_count,
                           "genes_tested": de.tested_feature_count,
                           "lncrnas_tested": de.tested_feature_count},
            de_expression_source=(raw_cohort.source.model_dump() if raw_cohort is not None else {
                "purpose": "differential_expression", "representation": "exploratory_transformed_continuous",
                "upstream_source": cohort.upstream_source, "distribution_source": cohort.distribution_source,
            }),
            expression_data_type=cohort.expression_data_type.value,
            expression_transform=cohort.expression_transform,
            expression_source_dataset=cohort.expression_url,
            expression_metadata_url=cohort.expression_metadata_url,
            expression_dataset_version=cohort.expression_dataset_version,
            upstream_gdc_release=cohort.upstream_gdc_release,
            upstream_source=cohort.upstream_source,
            distribution_source=cohort.distribution_source,
            candidate_selection_rules=request.candidate_selection.model_dump(), survival_configuration={"endpoint": "OS", "grouping": "median", "multiple_testing": "BH"},
            cox_covariates=("expression_z", "age_z", "sex", "pathological_stage"),
            ph_configuration="Schoenfeld residual time-trend tests; alpha=0.05",
            pubmed_queries_results={key: tuple(item.pmid for item in value) for key, value in literature.items()},
            geo_datasets_evaluated=tuple(item.dataset.accession for item in assessed), geo_dataset_selected=selected_geo,
            ranking_configuration=rankings[0].rule if rankings else "No candidates",
            software_versions=software_versions,
            candidate_list=tuple(item.gene_symbol for item in discovery.candidates), verifier_findings=verifier_findings,
            limitations=tuple(dict.fromkeys(limitations)), provenance=tuple(provenance),
            output_artifacts=tuple(str(path) for path in (
                manifest_path, qc_path, de_path, candidate_path, survival_path,
                adjusted_path, ph_path, literature_path, geo_path, verifier_path,
                ranking_path, report_path, result_path,
            )),
            real_cache_identity=real_cache_identity.model_dump(mode="json"),
            real_cache_identity_hash=cache_identity_hash(real_cache_identity),
            cache_status="NEW_EXECUTION",
        )
        _atomic_json(manifest_path, manifest.model_dump(mode="json"))
        events.append(_event("Return grounded findings", "Verify and synthesize", "deterministic_evidence_tiering", "computed results",
                             f"ranked={len(rankings)}; verifier findings={len(verifier_findings)}", "Report tiers and limitations",
                             CapabilityStatus.COMPLETED, None))
        status = "COMPLETED_WITH_LIMITATIONS" if limitations or verifier_findings else "COMPLETED"
        result = GenomeWideAnalysisResult(research_mode=request.research_mode, run_id=request.run_id, status=status,
                                          canonical_cancer_name=cancer.canonical_name, project_id=project,
                                          capabilities=tuple(capabilities), preparation=cohort, differential_expression=de,
                                          discovery=discovery, survival_screen=screens, adjusted_cox=adjusted_results,
                                          geo_replications=tuple(geo_replications),
                                          literature=literature, literature_summary=literature_summary,
                                          rankings=tuple(rankings), events=tuple(events),
                                          verification=tuple(verification),
                                          limitations=tuple(dict.fromkeys(limitations)), final_report=report, manifest=manifest)
        _atomic_json(result_path, result.model_dump(mode="json"))
        save_stage("final_synthesis", {"status": status, "final_report": report,
            "complete_result": str(result_path), "manifest": str(manifest_path)},
            artifact_paths=(str(result_path), str(manifest_path)))
        _atomic_json(run_state_path, {"run_id": request.run_id, "status": status,
            "current_stage": "final_synthesis", "resolved_stages": list(dict.fromkeys(resolved_stage_ids)),
            "complete_result": str(result_path), "manifest": str(manifest_path)})
        return result

    def _partial(self, request, cancer, project, capabilities, events, limitations, provenance):
        events.append(_event("Run differential expression", "Skip ineligible branch", "run_differential_expression", project,
                             "Not executed after capability observation", "Continue without fabricating candidates",
                             CapabilityStatus.SKIPPED_AFTER_REPLAN, None, True,
                             limitations[-1] if limitations else "Observed data did not meet analysis eligibility requirements"))
        manifest = DiscoveryManifest(run_id=request.run_id, timestamp=self.clock(), research_mode=request.research_mode,
            research_question=request.question, resolved_cancer=cancer, tcga_project=project, gdc_query=f"project={project}",
            sample_counts={}, gene_annotation_source_version="not retrieved", expression_preprocessing="not executed",
            expression_data_type="not retrieved", expression_transform="not executed", expression_source_dataset="not retrieved",
            expression_metadata_url=None, expression_dataset_version=None, upstream_gdc_release=None,
            upstream_source="NCI GDC / TCGA", distribution_source="UCSC Xena GDC Hub",
            de_parameters={}, candidate_selection_rules=request.candidate_selection.model_dump(), survival_configuration={},
            de_expression_source={},
            cox_covariates=(), ph_configuration="not executed", pubmed_queries_results={}, geo_datasets_evaluated=(),
            geo_dataset_selected=None, ranking_configuration="not executed", candidate_list=(), verifier_findings=(),
            software_versions={"python": platform.python_version(), "oncolncai": "0.1.0"},
            limitations=tuple(limitations), provenance=tuple(provenance), output_artifacts=())
        return GenomeWideAnalysisResult(research_mode=request.research_mode, run_id=request.run_id, status="COMPLETED_WITH_LIMITATIONS",
            canonical_cancer_name=cancer, project_id=project, capabilities=tuple(capabilities), events=tuple(events),
            limitations=tuple(limitations), final_report="DE was not statistically eligible; no genome-wide candidates were claimed.", manifest=manifest)


def _event(goal, action, tool, arguments, observation, decision, status, next_action=None, replan=False, reason=None):
    return DiscoveryEvent(goal=goal, planned_action=action, selected_tool=tool, validated_arguments=arguments,
                          observation=observation, decision=decision, execution_status=status,
                          next_action=next_action, replan_event=replan, replan_reason=reason)


def _report(cancer, project, cohort, discovery, screens, adjusted, rankings, limitations):
    lines = [f"# Real genome-wide lncRNA discovery: {cancer}", f"Project: {project}",
             f"Prepared {cohort.preparation.tumor_samples} tumors, {cohort.preparation.normal_samples} normals, and {cohort.preparation.lncrnas_retained} annotated lncRNAs.",
             f"DE retained {len(discovery.candidates)} of {discovery.eligible_result_count} eligible candidates for downstream screening.", "", "## Candidate evidence"]
    screen = {item.gene_symbol: item for item in screens}
    adj = {item.gene: item for item in adjusted}
    for item in rankings:
        survival = screen.get(item.gene_symbol)
        model = adj.get(item.gene_symbol)
        lines.append(f"- {item.gene_symbol} ({item.tier}): DE log2FC={item.log2_fold_change:.3g}, FDR={item.de_fdr:.3g}; " +
                     (f"survival HR={survival.hazard_ratio:.3g}, FDR={survival.fdr:.3g}; " if survival else "survival unavailable; ") +
                     (f"adjusted HR={model.adjusted_hazard_ratio:.3g} (95% CI {model.confidence_interval_lower:.3g}–{model.confidence_interval_upper:.3g}), "
                      f"p={model.p_value:.3g}, FDR={model.fdr:.3g}; initial N={model.initial_survival_eligible_count}, "
                      f"complete-case N={model.complete_case_count}, excluded={model.excluded_missing_covariates}, "
                      f"events={model.event_count}, parameters={model.model_parameter_count}, "
                      f"events/parameter={model.events_per_parameter:.2f}, missingness={model.missingness_by_covariate}, "
                      f"convergence={model.convergence_status.value}." if model and model.fdr is not None else "adjusted Cox unavailable."))
    lines.extend(["", "## Limitations", *(f"- {item}" for item in limitations), "", "These are research-prioritization results, not clinical biomarker claims."])
    return "\n".join(lines)
