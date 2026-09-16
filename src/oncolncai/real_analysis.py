"""Real-data candidate validation composed from existing deterministic tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, ConfigDict, Field

from oncolncai.cancer_resolver import CancerResolveRequest, ResolutionStatus, resolve_cancer
from oncolncai.pubmed import PubMedClient, PubMedRecord, PubMedSearchRequest
from oncolncai.schemas import Provenance, ResultStatus
from oncolncai.survival import SurvivalAnalysisResult, run_survival_analysis
from oncolncai.tcga_capabilities import GDCCapabilityClient, TCCapabilityRequest, DataCapabilities
from oncolncai.tcga_survival import TCGASurvivalDataClient, TCGASurvivalDataRequest
from oncolncai.verifier import (
    ClaimKind,
    EvidenceReference,
    NumericAssertion,
    VerificationClaim,
    VerificationReport,
    VerificationRequest,
    verify_claims,
)


class RealRunStatus(StrEnum):
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_LIMITATIONS = "COMPLETED_WITH_LIMITATIONS"
    FAILED = "FAILED"


class WorkflowStepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED_AFTER_REPLAN = "skipped_after_replan"
    FAILED = "failed"


class WorkflowStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    label: str
    status: WorkflowStepStatus
    tool: str
    input_summary: str
    observation: str
    decision: str
    next_action: str | None = None
    replanned: bool = False
    replan_reason: str | None = None
    provenance: tuple[Provenance, ...] = ()


class CandidateRanking(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    evidence_tier: str
    literature_sources: int = Field(ge=0)
    survival_significant: bool | None = None
    ph_assumption_met: bool | None = None
    external_validation: str = "NOT AVAILABLE"
    rationale: str


class AnalysisManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_version: str = "real-candidate-validation-v1"
    run_id: str
    timestamp: datetime
    question: str
    cancer: str
    project_id: str
    candidate: str
    ensembl_gene_id: str
    sample_count: int | None = None
    event_count: int | None = None
    endpoint: str = "OS"
    grouping: str = "median"
    alpha: float = 0.05
    literature_pmids: tuple[str, ...] = ()
    provenance: tuple[Provenance, ...] = ()


class RealAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    question: str = Field(min_length=1, max_length=2000)
    cancer_name: str = Field(min_length=1, max_length=200)
    candidate: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9._-]*$")
    ensembl_gene_id: str = Field(pattern=r"^ENSG\d{11}$")
    max_pubmed_results: int = Field(default=10, ge=1, le=20)


class RealAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_mode: str = "REAL DATA"
    run_id: str
    status: RealRunStatus
    question: str
    canonical_cancer_name: str
    project_id: str
    candidate: str
    capabilities: DataCapabilities | None = None
    papers: tuple[PubMedRecord, ...] = ()
    survival: SurvivalAnalysisResult | None = None
    ranking: CandidateRanking
    verification: VerificationReport | None = None
    workflow: tuple[WorkflowStep, ...]
    limitations: tuple[str, ...]
    provenance: tuple[Provenance, ...]
    manifest: AnalysisManifest
    final_report: str


ProgressCallback = Callable[[float, str], None]


class RealAnalysisRunner:
    """Run a real, provenance-bearing single-candidate TCGA validation path."""

    def __init__(
        self,
        *,
        pubmed_client: PubMedClient | None = None,
        capability_client: GDCCapabilityClient | None = None,
        survival_client: TCGASurvivalDataClient | None = None,
        clock: Callable[[], datetime] | None = None,
        manifest_directory: Path | None = None,
    ) -> None:
        self._pubmed = pubmed_client or PubMedClient()
        self._capabilities = capability_client or GDCCapabilityClient()
        self._survival = survival_client or TCGASurvivalDataClient()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._manifest_directory = manifest_directory

    def run(
        self, request: RealAnalysisRequest, *, progress: ProgressCallback | None = None
    ) -> RealAnalysisResult:
        update = progress or (lambda _fraction, _message: None)
        steps: list[WorkflowStep] = []
        limitations: list[str] = []
        provenance: list[Provenance] = []

        update(0.05, "Step 1/9 · resolve_cancer")
        resolved_result = resolve_cancer(CancerResolveRequest(name=request.cancer_name))
        resolved = resolved_result.data
        if resolved is None or resolved.status != ResolutionStatus.RESOLVED or not resolved.tcga_project_id or not resolved.canonical_name:
            raise ValueError("REAL DATA mode requires an unambiguous supported TCGA cancer")
        provenance.extend(resolved_result.provenance)
        steps.append(self._step("resolve", "Cancer resolution", "resolve_cancer", request.cancer_name, f"Resolved {resolved.canonical_name} to {resolved.tcga_project_id}.", "Use the resolved project for every downstream tool.", "capabilities", resolved_result.provenance))

        update(0.14, "Step 2/9 · inspect_tcga_capabilities")
        capability_result = self._capabilities.inspect(TCCapabilityRequest(project_id=resolved.tcga_project_id))
        capabilities = capability_result.data if capability_result.status != ResultStatus.FAILURE else None
        if capabilities is None:
            limitations.append("TCGA capability inspection failed: " + (capability_result.error.message if capability_result.error else "unknown error"))
            steps.append(self._failed_step("capabilities", "Dataset capability", "inspect_tcga_capabilities", resolved.tcga_project_id, limitations[-1], "survival"))
        else:
            provenance.extend(capability_result.provenance)
            steps.append(self._step("capabilities", "Dataset capability", "inspect_tcga_capabilities", resolved.tcga_project_id, f"Found {capabilities.tumor_samples} tumors, {capabilities.normal_samples} normals and {capabilities.survival_samples} survival-usable cases.", "Continue with supported real analyses.", "survival", capability_result.provenance))

        update(0.24, "Step 3/9 · retrieve_expression_and_clinical_data")
        prepared = self._survival.prepare(TCGASurvivalDataRequest(project_id=resolved.tcga_project_id, gene=request.candidate, ensembl_gene_id=request.ensembl_gene_id))
        survival = None
        if prepared.status == ResultStatus.FAILURE or prepared.data is None:
            message = prepared.error.message if prepared.error else "real TCGA cohort preparation failed"
            limitations.append("Real TCGA survival preparation unavailable: " + message)
            steps.append(self._failed_step("survival", "Real TCGA survival", "prepare_tcga_survival_cohort", f"{request.candidate} ({request.ensembl_gene_id})", message, "literature"))
        else:
            provenance.extend(prepared.provenance)
            update(0.48, "Step 4/9 · survival_analysis")
            analyzed = run_survival_analysis(prepared.data.survival_request())
            if analyzed.status == ResultStatus.FAILURE or analyzed.data is None:
                message = analyzed.error.message if analyzed.error else "survival model failed"
                limitations.append("Survival analysis unavailable: " + message)
                steps.append(self._failed_step("survival", "Real TCGA survival", "run_survival_analysis", request.candidate, message, "literature"))
            else:
                survival = analyzed.data
                provenance.extend(analyzed.provenance)
                limitations.extend(analyzed.warnings)
                steps.append(self._step("survival", "Real TCGA survival", "run_survival_analysis", f"{request.candidate}; OS; median split", f"n={survival.sample_count}, events={survival.event_count}, HR={survival.hazard_ratio:.3f}, Cox p={survival.cox_p_value:.3g}, log-rank p={survival.log_rank_p_value:.3g}.", "Retain the deterministic result and PH diagnostic.", "literature", analyzed.provenance))

        update(0.62, "Step 5/9 · search_pubmed")
        literature = self._pubmed.search(PubMedSearchRequest(cancer=resolved.canonical_name, concepts=(request.candidate,), max_results=request.max_pubmed_results))
        retrieved_papers = literature.data.records if literature.status != ResultStatus.FAILURE and literature.data else ()
        papers = tuple(
            paper
            for paper in retrieved_papers
            if _is_contextually_relevant(
                paper,
                candidate=request.candidate,
                cancer_terms=(resolved.canonical_name, resolved.tcga_project_id.removeprefix("TCGA-")),
            )
        )
        if papers:
            provenance.extend(literature.provenance)
            steps.append(self._step("literature", "Published evidence", "search_pubmed", f"{resolved.canonical_name} AND {request.candidate}", f"Retrieved {len(retrieved_papers)} record(s); retained {len(papers)} after deterministic cancer/candidate context filtering.", "Retain PMID metadata; do not infer unsupported article claims.", "external_validation", literature.provenance))
        else:
            message = literature.error.message if literature.error else "no PubMed records were retrieved"
            limitations.append("Literature evidence unavailable: " + message)
            steps.append(self._failed_step("literature", "Published evidence", "search_pubmed", request.candidate, message, "external_validation"))

        update(0.72, "Step 6/9 · replan_unsupported_branches")
        de_reason = "Real count-matrix DE preparation is not implemented in the deployed path."
        if capabilities is not None and not capabilities.de_recommended:
            de_reason = "TCGA differential expression was not appropriate under the observed sample-count thresholds."
        limitations.append(de_reason)
        steps.append(self._skipped("de", "Differential expression", "run_differential_expression", de_reason, "external_validation"))
        external_reason = "No compatible cohort was processed; GEO expression replication is not implemented."
        limitations.append("External validation: NOT AVAILABLE — " + external_reason)
        steps.append(self._skipped("external_validation", "External validation", "search_geo", external_reason, "verification"))
        limitations.append("Multivariable Cox was not evaluated because harmonized age, sex and stage inputs are not available in this path.")

        update(0.82, "Step 7/9 · rank_candidate")
        significant = survival.cox_p_value < 0.05 if survival else None
        ph_met = survival.diagnostics.proportional_hazards_assumption_met if survival else None
        tier = "Tier B" if significant and ph_met and len(papers) >= 2 else "Tier C" if survival or papers else "Tier D"
        rationale = "Evidence is preliminary or incomplete; no independent replication was performed."
        ranking = CandidateRanking(candidate=request.candidate, evidence_tier=tier, literature_sources=len(papers), survival_significant=significant, ph_assumption_met=ph_met, rationale=rationale)
        steps.append(self._step("ranking", "Candidate ranking", "deterministic_evidence_tiering", request.candidate, f"Assigned {tier} from available evidence and missingness.", rationale, "verification", ()))

        update(0.90, "Step 8/9 · verify_results")
        verification = self._verify(resolved.tcga_project_id, request.candidate, survival)
        if verification is not None and verification.passed:
            steps.append(self._step("verification", "Verifier", "verify_claims", request.candidate, "Numeric consistency, identifier, provenance and causal-language checks passed.", "Permit restrained synthesis.", "final_report", tuple(provenance)))
        else:
            limitations.append("Verifier did not approve a computed survival claim.")
            steps.append(self._failed_step("verification", "Verifier", "verify_claims", request.candidate, limitations[-1], "final_report"))

        useful = survival is not None or bool(papers)
        limited = bool(limitations) or verification is None or not verification.passed
        status = RealRunStatus.FAILED if not useful else RealRunStatus.COMPLETED_WITH_LIMITATIONS if limited else RealRunStatus.COMPLETED
        update(0.97, "Step 9/9 · synthesize_final_report")
        report = self._report(request, resolved.canonical_name, resolved.tcga_project_id, papers, survival, ranking, limitations)
        steps.append(self._step("final_report", "Final synthesis", "deterministic_grounded_report", request.question, "Produced a report from retrieved metadata and deterministic outputs.", f"Return {status.value}.", None, tuple(provenance)))
        manifest = AnalysisManifest(run_id=request.run_id, timestamp=self._clock(), question=request.question, cancer=resolved.canonical_name, project_id=resolved.tcga_project_id, candidate=request.candidate, ensembl_gene_id=request.ensembl_gene_id, sample_count=survival.sample_count if survival else None, event_count=survival.event_count if survival else None, literature_pmids=tuple(p.pmid for p in papers), provenance=tuple(provenance))
        if self._manifest_directory is not None:
            save_analysis_manifest(self._manifest_directory, manifest)
        update(1.0, f"{status.value} · workflow finished")
        return RealAnalysisResult(run_id=request.run_id, status=status, question=request.question, canonical_cancer_name=resolved.canonical_name, project_id=resolved.tcga_project_id, candidate=request.candidate, capabilities=capabilities, papers=papers, survival=survival, ranking=ranking, verification=verification, workflow=tuple(steps), limitations=tuple(dict.fromkeys(limitations)), provenance=tuple(provenance), manifest=manifest, final_report=report)

    @staticmethod
    def _step(step_id: str, label: str, tool: str, inputs: str, observation: str, decision: str, next_action: str | None, provenance: tuple | list) -> WorkflowStep:
        return WorkflowStep(step_id=step_id, label=label, status=WorkflowStepStatus.COMPLETED, tool=tool, input_summary=inputs, observation=observation, decision=decision, next_action=next_action, provenance=tuple(provenance))

    @staticmethod
    def _failed_step(step_id: str, label: str, tool: str, inputs: str, reason: str, next_action: str | None) -> WorkflowStep:
        return WorkflowStep(step_id=step_id, label=label, status=WorkflowStepStatus.FAILED, tool=tool, input_summary=inputs, observation=reason, decision="Preserve completed work and continue where scientifically valid.", next_action=next_action)

    @staticmethod
    def _skipped(step_id: str, label: str, tool: str, reason: str, next_action: str | None) -> WorkflowStep:
        return WorkflowStep(step_id=step_id, label=label, status=WorkflowStepStatus.SKIPPED_AFTER_REPLAN, tool=tool, input_summary="Not executed", observation=reason, decision="Skip this branch and retain supported analyses.", next_action=next_action, replanned=True, replan_reason=reason)

    @staticmethod
    def _verify(project_id: str, candidate: str, survival: SurvivalAnalysisResult | None) -> VerificationReport | None:
        if survival is None:
            return None
        text = f"In {project_id}, {candidate} high versus low expression had HR {survival.hazard_ratio:.6g} and Cox p {survival.cox_p_value:.6g}."
        reference = EvidenceReference(reference_id=f"survival:{project_id}:{candidate}", source_type="survival_analysis", source_text=text, gene_identifier=candidate, numeric_values={"hazard_ratio": survival.hazard_ratio, "cox_p_value": survival.cox_p_value, "sample_count": float(survival.sample_count), "event_count": float(survival.event_count)}, provenance=(Provenance(source="TCGA deterministic survival", source_id=project_id),))
        claim = VerificationClaim(claim_id="survival-claim", text=text, kind=ClaimKind.SURVIVAL, gene_identifier=candidate, evidence_reference_ids=(reference.reference_id,), numeric_assertions=(NumericAssertion(evidence_reference_id=reference.reference_id, metric="hazard_ratio", reported_value=survival.hazard_ratio), NumericAssertion(evidence_reference_id=reference.reference_id, metric="cox_p_value", reported_value=survival.cox_p_value)))
        result = verify_claims(VerificationRequest(claims=(claim,), evidence=(reference,)))
        return result.data

    @staticmethod
    def _report(request: RealAnalysisRequest, cancer: str, project_id: str, papers: tuple[PubMedRecord, ...], survival: SurvivalAnalysisResult | None, ranking: CandidateRanking, limitations: list[str]) -> str:
        lines = [f"# Real-data candidate report: {request.candidate}", f"Cancer: {cancer} ({project_id})", "", "## COMPUTED EVIDENCE"]
        if survival:
            ph = survival.diagnostics.proportional_hazards_assumption_met
            lines.append(f"Overall survival median split: n={survival.sample_count}, events={survival.event_count}, HR={survival.hazard_ratio:.3f} (95% CI {survival.confidence_interval_lower:.3f}–{survival.confidence_interval_upper:.3f}), Cox p={survival.cox_p_value:.3g}, log-rank p={survival.log_rank_p_value:.3g}; PH assumption met={ph}.")
        else:
            lines.append("Survival analysis unavailable.")
        lines.extend(["", "## PUBLISHED EVIDENCE"])
        lines.extend([f"- PMID {p.pmid} ({p.publication_year or 'year unavailable'}): {p.title}" for p in papers] or ["Literature evidence unavailable."])
        lines.extend(["", "## EXTERNAL VALIDATION", "NOT AVAILABLE", "", "## CANDIDATE RANKING", f"{ranking.evidence_tier}: {ranking.rationale}", "", "## LIMITATIONS"])
        lines.extend(f"- {item}" for item in limitations)
        lines.extend(["", "## CONCLUSION", f"{request.candidate} is a computational biomarker candidate with {ranking.evidence_tier.lower()} evidence in this run. No clinical validity or utility is established."])
        return "\n".join(lines)


def save_analysis_manifest(directory: Path, manifest: AnalysisManifest) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{manifest.run_id}.manifest.json"
    payload = manifest.model_dump_json(indent=2)
    with NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as temporary:
        temporary.write(payload + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(destination)
    return destination


def _is_contextually_relevant(
    paper: PubMedRecord, *, candidate: str, cancer_terms: tuple[str, ...]
) -> bool:
    """Reject lexical gene-name collisions before displaying a paper as cancer evidence."""

    text = f"{paper.title} {paper.abstract or ''}".casefold()
    text = text.replace("metastasis-associated lung adenocarcinoma transcript 1", "")
    text = text.replace("metastasis associated lung adenocarcinoma transcript 1", "")
    if candidate.casefold() not in text:
        return False
    terms = {
        term.casefold().strip()
        for term in cancer_terms
        if len(term.strip()) >= 4 and term.casefold().strip() not in {"cancer", "carcinoma"}
    }
    return any(term in text for term in terms)
