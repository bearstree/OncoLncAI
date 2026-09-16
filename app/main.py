"""Gradio service for structured real and synthetic OncoLncAI runs."""

from __future__ import annotations

import os
import re
import math
import threading
import time
from datetime import datetime, timezone

import gradio as gr

from oncolncai.config import load_settings
from oncolncai.cancer_resolver import (
    CancerResolveRequest, ResolutionStatus, cancer_from_display_name,
    cancers_mentioned_in_text, list_supported_cancers, resolve_cancer,
)
from oncolncai.demo import OfflineDemoReport, run_configured_agent_demo
from oncolncai.discovery_analysis import GenomeWideAnalysisRequest, GenomeWideAnalysisResult, GenomeWideAnalysisRunner
from oncolncai.gdc_counts import GDCRawCountClient
from oncolncai.pubmed import PubMedClient
from oncolncai.real_analysis import RealAnalysisRequest, RealAnalysisResult, RealAnalysisRunner
from oncolncai.tcga_capabilities import GDCCapabilityClient
from oncolncai.tcga_discovery import ResearchMode, TCGAGenomeWideClient
from oncolncai.tcga_survival import TCGASurvivalDataClient
from oncolncai.ui import build_demo_run_view, build_genome_wide_run_view, build_real_run_view
from oncolncai.workflow_stages import WORKFLOW_STAGES
from oncolncai.workflow_stages import StageCheckpoint


APP_CSS = """
.hero {max-width: 1200px; margin: auto}.result-card {border:1px solid #cbdde4;border-radius:10px;padding:12px 18px}
.run-status {border-left:8px solid #177e89;background:#f4fafb}.volcano svg{width:100%;max-height:430px;background:#fff}
"""

RESEARCH_MODE_CHOICES = [
    ("Genome-Wide Discovery", ResearchMode.GENOME_WIDE_DISCOVERY.value),
    ("Targeted Candidate Validation", ResearchMode.SINGLE_CANDIDATE_VALIDATION.value),
]
CANCER_CHOICES = [item.display_name for item in list_supported_cancers()]

GENE_IDENTIFIER_PAIRS = {"NEAT1": "ENSG00000245532"}
_ACTIVE_RUN = threading.Lock()
IDLE_STATUS_TEXT = "**Ready to run** · REAL DATA · Genome-Wide Discovery  \nTCGA project will be resolved when analysis starts."


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")


def run_selected_analysis(mode: str, question: str, cancer_name: str, candidate: str,
                          ensembl_gene_id: str, research_mode: str = "GENOME_WIDE_DISCOVERY",
                          progress: gr.Progress = gr.Progress(), requested_run_id: str | None = None,
                          checkpoint_callback=None) -> dict:
    """Execute a backend workflow and retain timing/configuration metadata."""

    started = datetime.now(timezone.utc)
    tick = time.monotonic()
    settings = load_settings()
    cancer_name = validate_question_and_cancer(question, cancer_name)
    candidate, ensembl_gene_id = validate_candidate_inputs(research_mode, candidate, ensembl_gene_id)
    assigned_run_id = requested_run_id or _run_id()
    metadata = {"started_at": started.isoformat(), "run_id": assigned_run_id, "agent": "OncoLncAI Supervisor",
                "model": f"{settings.llm_provider}/{settings.llm_model}", "research_mode": research_mode}
    try:
        if mode == "DEMO / SYNTHETIC":
            progress(.02, desc="Starting synthetic control-flow validation")
            report = run_configured_agent_demo(cancer_name, settings,
                progress_callback=lambda fraction, message: progress(fraction, desc=message))
            payload = report.model_dump(mode="json")
        else:
            api_key = settings.ncbi_api_key.get_secret_value() if settings.ncbi_api_key else None
            if research_mode == ResearchMode.GENOME_WIDE_DISCOVERY.value:
                runner = GenomeWideAnalysisRunner(
                    genome_client=TCGAGenomeWideClient(cache_dir=settings.cache_dir / "tcga_discovery"),
                    capability_client=GDCCapabilityClient(cache_dir=settings.cache_dir / "gdc"),
                    survival_client=TCGASurvivalDataClient(cache_dir=settings.cache_dir / "tcga_survival"),
                    pubmed_client=PubMedClient(api_key=api_key, cache_dir=settings.cache_dir / "pubmed"),
                    output_directory=settings.cache_dir / "runs",
                    raw_count_client=GDCRawCountClient(cache_dir=settings.cache_dir / "gdc_raw_counts"))
                payload = runner.run(
                    GenomeWideAnalysisRequest(run_id=assigned_run_id, question=question, cancer_name=cancer_name),
                    progress=lambda snapshot: progress(snapshot.fraction, desc=snapshot.message),
                    checkpoint=checkpoint_callback,
                ).model_dump(mode="json")
            else:
                runner = RealAnalysisRunner(
                    pubmed_client=PubMedClient(api_key=api_key, cache_dir=settings.cache_dir / "pubmed"),
                    capability_client=GDCCapabilityClient(cache_dir=settings.cache_dir / "gdc"),
                    survival_client=TCGASurvivalDataClient(cache_dir=settings.cache_dir / "tcga_survival"),
                    manifest_directory=settings.cache_dir / "manifests")
                payload = runner.run(RealAnalysisRequest(run_id=assigned_run_id, question=question, cancer_name=cancer_name,
                                                         candidate=candidate, ensembl_gene_id=ensembl_gene_id),
                                     progress=lambda f, message: progress(f, desc=message)).model_dump(mode="json")
        metadata["elapsed_seconds"] = round(time.monotonic() - tick, 2)
        return {"analysis_mode": mode, "research_mode": research_mode, "result": payload, "run_metadata": metadata}
    except Exception as exc:  # returned to the status card instead of leaving an indeterminate UI
        metadata["elapsed_seconds"] = round(time.monotonic() - tick, 2)
        return {"analysis_mode": mode, "research_mode": research_mode, "error": str(exc), "run_metadata": metadata}


def run_demo(cancer_name: str) -> dict:
    return run_configured_agent_demo(cancer_name, load_settings()).model_dump(mode="json")


def runtime_configuration() -> dict[str, object]:
    settings = load_settings()
    return {"environment": settings.environment, "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model, "llm_base_url": settings.llm_base_url,
            "llm_timeout_seconds": settings.llm_timeout_seconds,
            "llm_api_key_configured": settings.llm_api_key is not None,
            "ncbi_api_key_configured": settings.ncbi_api_key is not None}


def validate_candidate_inputs(research_mode: str, gene_symbol: str, ensembl_id: str) -> tuple[str, str]:
    """Clear genome-wide identifiers and validate supported single-candidate pairs."""

    if research_mode == ResearchMode.GENOME_WIDE_DISCOVERY.value:
        return "", ""
    symbol, ensembl = gene_symbol.strip().upper(), ensembl_id.strip().upper()
    reverse = {value: key for key, value in GENE_IDENTIFIER_PAIRS.items()}
    if not symbol and ensembl in reverse:
        symbol = reverse[ensembl]
    if not ensembl and symbol in GENE_IDENTIFIER_PAIRS:
        ensembl = GENE_IDENTIFIER_PAIRS[symbol]
    if not symbol or not ensembl:
        raise ValueError("Single-candidate validation requires a resolvable gene symbol or Ensembl ID; this backend needs both resolved identifiers.")
    expected = GENE_IDENTIFIER_PAIRS.get(symbol)
    if expected is not None and expected != ensembl:
        raise ValueError(f"Gene symbol {symbol} conflicts with Ensembl ID {ensembl}; expected {expected}.")
    reverse_symbol = reverse.get(ensembl)
    if reverse_symbol is not None and reverse_symbol != symbol:
        raise ValueError(f"Ensembl ID {ensembl} conflicts with gene symbol {symbol}; expected {reverse_symbol}.")
    return symbol, ensembl


def validate_question_and_cancer(question: str, selection: str) -> str:
    """Return the canonical cancer or reject an explicit question/project conflict."""

    try:
        selected = cancer_from_display_name(selection)
    except ValueError:
        resolved = resolve_cancer(CancerResolveRequest(name=selection)).data
        if resolved is None or resolved.status != ResolutionStatus.RESOLVED or not resolved.tcga_project_id:
            raise ValueError("Select a supported, unambiguous TCGA cancer project.")
        selected = cancer_from_display_name(resolved.tcga_project_id)
    mentioned = cancers_mentioned_in_text(question)
    conflicts = [item for item in mentioned if item.tcga_project_id != selected.tcga_project_id]
    if conflicts:
        names = ", ".join(item.display_name for item in conflicts)
        raise ValueError(f"Research question mentions {names}, but the selected project is {selected.display_name}. Resolve this mismatch before analysis.")
    return selected.canonical_name


def candidate_input_state(research_mode: str) -> tuple[object, object, object]:
    """Dynamically reveal candidate controls and clear values when hidden."""

    visible = research_mode == ResearchMode.SINGLE_CANDIDATE_VALIDATION.value
    return gr.update(visible=visible), gr.update(value="" if not visible else None), gr.update(value="" if not visible else None)


RENDER_OUTPUT_COUNT = 21
DATAFRAME_OUTPUT_INDICES = frozenset({1, 5, 7, 8, 9, 11, 13, 14, 15, 18})
FILE_OUTPUT_INDEX = 20


def empty_render_values() -> tuple[object, ...]:
    """Return component-compatible empty values in canonical renderer order."""
    values: list[object] = [""] * RENDER_OUTPUT_COUNT
    for index in DATAFRAME_OUTPUT_INDICES:
        values[index] = []
    values[FILE_OUTPUT_INDEX] = None
    return tuple(values)


def normalize_render_values(values: tuple[object, ...] | list[object]) -> tuple[object, ...]:
    """Enforce the heterogeneous Gradio output contract before every yield."""
    if len(values) != RENDER_OUTPUT_COUNT:
        raise ValueError(f"renderer returned {len(values)} values; expected {RENDER_OUTPUT_COUNT}")
    normalized = list(values)
    for index in DATAFRAME_OUTPUT_INDICES:
        value = normalized[index]
        if value is None or (isinstance(value, str) and not value.strip()):
            normalized[index] = []
    file_value = normalized[FILE_OUTPUT_INDEX]
    if isinstance(file_value, str) and not file_value.strip():
        normalized[FILE_OUTPUT_INDEX] = None
    return tuple(normalized)


EMPTY = empty_render_values()


def checkpoint_volcano_html(rows: list[dict]) -> str:
    """Render checkpointed DE statistics without recomputing any values."""
    if not rows:
        return "<p>DE completed with zero testable results.</p>"
    max_x = max(1.0, max(abs(float(row.get("log2_fold_change") or 0)) for row in rows))
    ys = [-math.log10(max(float(row.get("adjusted_p_value") or 1), 1e-300)) for row in rows]
    max_y = max(1.0, max(ys))
    circles = []
    for row, y_value in zip(rows, ys):
        x = 400 + float(row.get("log2_fold_change") or 0) / max_x * 350
        y = 350 - y_value / max_y * 315
        color = "#d97706" if row.get("significant") else "#94a3b8"
        label = row.get("gene_symbol") or row.get("feature_id") or "feature"
        circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="{color}"><title>{label}; log2FC={row.get("log2_fold_change")}; FDR={row.get("adjusted_p_value")}</title></circle>')
    return ('<div class="volcano"><svg viewBox="0 0 800 390" role="img" aria-label="Checkpointed DE volcano plot">'
            '<line x1="40" y1="350" x2="770" y2="350" stroke="#64748b"/>'
            '<line x1="400" y1="25" x2="400" y2="350" stroke="#cbd5e1"/>'
            + "".join(circles) + '<text x="350" y="380">log2 fold change</text></svg></div>')


def apply_stage_checkpoint(values: tuple[object, ...] | list[object],
                           checkpoint: StageCheckpoint) -> tuple[object, ...]:
    """Merge one completed stage into the retained progressive UI projection."""
    rendered = list(normalize_render_values(values))
    payload, stage, status = checkpoint.payload, checkpoint.stage_id, checkpoint.status
    if stage == "data_qc":
        prep = payload.get("preparation", {})
        rendered[2] = (f"**{status}** · project `{payload.get('project_id', 'Unknown')}`  \n"
                       f"Retrieved {prep.get('source_columns', 'Not evaluated')} samples; selected "
                       f"{prep.get('tumor_samples', 'Not evaluated')} primary tumors and "
                       f"{prep.get('normal_samples', 'Not evaluated')} solid-tissue normals. "
                       f"Annotated lncRNAs: {prep.get('lncrnas_retained', 'Not evaluated')}.")
    elif stage == "differential_expression":
        config, rows = payload.get("configuration", {}), payload.get("results", [])
        rendered[3] = (f"**{status}** · tumor N={payload.get('case_sample_count')} · normal N={payload.get('control_sample_count')} · "
                       f"tested={payload.get('tested_feature_count')} · method={config.get('method')} · "
                       f"multiple testing={config.get('multiple_testing_method')}")
        rendered[4] = checkpoint_volcano_html(rows)
        rendered[5] = [[row.get("gene_symbol") or "", row.get("feature_id"), "lncRNA",
                        row.get("case_mean_cpm"), row.get("control_mean_cpm"), row.get("log2_fold_change"),
                        row.get("p_value"), row.get("adjusted_p_value"),
                        "PASS" if row.get("significant") else "NOT SIGNIFICANT"] for row in rows]
    elif stage == "candidate_discovery":
        rendered[6] = (f"**{status}** · DE-qualified: {payload.get('eligible_result_count', 0)} · "
                       f"forwarded: {len(payload.get('candidates', []))}. Deterministic configured selection.")
    elif stage == "survival_analysis":
        rendered[7] = [[row.get("gene_symbol"), row.get("sample_count"), row.get("event_count"),
                        row.get("hazard_ratio"), f"{row.get('confidence_interval_lower')}–{row.get('confidence_interval_upper')}",
                        row.get("p_value"), row.get("fdr"), "median", status]
                       for row in payload.get("results", [])]
    elif stage == "adjusted_cox":
        rendered[8] = [[row.get("gene"), row.get("adjusted_hazard_ratio"),
                        f"{row.get('confidence_interval_lower')}–{row.get('confidence_interval_upper')}",
                        row.get("p_value"), row.get("fdr"), row.get("initial_survival_eligible_count"),
                        row.get("complete_case_count"), row.get("event_count"), row.get("missingness_by_covariate"),
                        ", ".join(row.get("covariates_used", [])), row.get("convergence_status"),
                        "FAIL" if row.get("overall_ph_violation") else "PASS"]
                       for row in payload.get("results", [])]
    elif stage == "ph_diagnostics":
        rendered[9] = [[row.get("gene"), row.get("term"), row.get("statistic"), row.get("p_value"),
                        "FAIL" if row.get("violation") else "PASS"] for row in payload.get("results", [])]
    elif stage == "literature_evidence":
        summary = payload.get("summary", {})
        rendered[10] = f"**{status}** · retrieved {summary.get('records_retrieved', 0)} records."
        rendered[11] = [[gene, paper.get("pmid"), paper.get("title"), paper.get("publication_year") or "Unavailable",
                         "Current cancer", "Retrieved article", "Not extracted", "Not extracted", "Not extracted",
                         "Not assessed", paper.get("query"), paper.get("retrieval_timestamp")]
                        for gene, papers in payload.get("records", {}).items() for paper in papers]
    elif stage == "geo_validation":
        rendered[12] = f"**{status}** — {payload.get('reason', 'No reason recorded')}"
        rendered[13] = [[row.get("accession"), row.get("platform"), row.get("tumor_count"), row.get("normal_count"),
                         ", ".join(row.get("mapped_candidates", [])), ", ".join(row.get("unmapped_candidates", [])),
                         "EXPRESSION REPLICATION", status] for row in payload.get("replications", [])]
    elif stage == "evidence_ranking":
        rendered[14] = [[index, row.get("gene_symbol"), row.get("ensembl_gene_id"),
                         "UP" if (row.get("log2_fold_change") or 0) > 0 else "DOWN",
                         row.get("log2_fold_change"), row.get("de_fdr"), "Not evaluated", row.get("survival_fdr"),
                         "Not evaluated", row.get("adjusted_cox_fdr"), "Not evaluated", len(row.get("literature_pmids", [])),
                         row.get("geo_expression_replication"), "Not evaluated", "Pending verification", row.get("tier"), "lncRNA"]
                        for index, row in enumerate(payload.get("results", []), 1)]
    elif stage == "verification":
        rendered[15] = [[row.get("check"), row.get("decision"), row.get("detail")]
                        for row in payload.get("results", [])]
    elif stage == "final_synthesis":
        rendered[17] = payload.get("final_report", "")
        rendered[20] = list(checkpoint.artifact_paths) or None
    return normalize_render_values(rendered)


def render_selected(payload: dict) -> tuple[object, ...]:
    """Render only values present in the structured backend response."""

    meta = payload.get("run_metadata", {})
    if "error" in payload:
        header = (f"## FAILED\n\n**Mode:** {payload.get('analysis_mode')} · **Research mode:** `{payload.get('research_mode')}`  \n"
                  f"**Start:** {meta.get('started_at', 'NA')} · **Elapsed:** {meta.get('elapsed_seconds', 'NA')} s · **Current stage:** Failed  \n"
                  f"**Reason:** {payload['error']}")
        values = list(EMPTY); values[0] = header; values[16] = f"- Run failed: {payload['error']}"
        return normalize_render_values(values)
    mode = payload["analysis_mode"]
    if mode == "REAL DATA" and payload.get("research_mode") == ResearchMode.GENOME_WIDE_DISCOVERY.value:
        view = build_genome_wide_run_view(GenomeWideAnalysisResult.model_validate(payload["result"]))
        overview = view.overview + f"  \n**Agent/model:** {meta.get('agent')} · `{meta.get('model')}` · **Elapsed:** {meta.get('elapsed_seconds')} s"
        return normalize_render_values((overview, [list(x) for x in view.workflow_rows], view.qc, view.de_summary, view.volcano_html,
                [list(x) for x in view.de_rows], view.discovery, [list(x) for x in view.survival_rows],
                [list(x) for x in view.adjusted_rows], [list(x) for x in view.ph_rows],
                view.literature_summary, [list(x) for x in view.literature_rows], view.geo_summary, [list(x) for x in view.geo_rows],
                [list(x) for x in view.ranking_rows], [list(x) for x in view.verifier_rows], view.limitations,
                view.report, [list(x) for x in view.provenance_rows], view.manifest, list(view.artifacts)))
    if mode == "REAL DATA":
        view = build_real_run_view(RealAnalysisResult.model_validate(payload["result"]))
        overview = view.header + f"  \n**Agent/model:** {meta.get('agent')} · `{meta.get('model')}` · **Elapsed:** {meta.get('elapsed_seconds')} s · **Current stage:** Final synthesis"
        values = list(EMPTY)
        values[0:4] = [overview, [list(x) for x in view.workflow_rows], view.qc, "Single-candidate mode does not run genome-wide DE."]
        values[7] = [[payload["result"]["candidate"], payload["result"].get("survival", {}).get("sample_count", "NA"),
                      payload["result"].get("survival", {}).get("event_count", "NA"), "See summary", "See summary", "See summary", "NA", "median", "COMPLETED"]] if payload["result"].get("survival") else []
        paper_count = len(view.literature_rows)
        values[10] = f"**PubMed records retrieved:** {paper_count} · **Unique:** {len({x[0] for x in view.literature_rows})} · **Candidate supported:** {int(paper_count > 0)} / 1"
        values[11] = [[payload["result"]["candidate"], *x, "Retrieved article", "", ""] for x in view.literature_rows]
        values[13] = []; values[14] = [list(x) for x in view.ranking_rows]
        values[15] = [[payload["result"]["candidate"], "ACCEPT" if "ACCEPT" in view.verifier else "REJECT", view.verifier]]
        values[16:21] = [view.limitations, view.report, [], view.manifest, []]
        return normalize_render_values(values)
    report = OfflineDemoReport.model_validate(payload["result"])
    view = build_demo_run_view("Synthetic capability-validation workflow", report)
    values = list(EMPTY)
    values[0] = view.overview + f"  \n**Final status:** COMPLETED · **Elapsed:** {meta.get('elapsed_seconds')} s"
    values[1] = [[step, status.upper().replace("_", " "), "control-flow fixture", "demo", "validated", "synthetic observation", "control-flow decision", "", ""] for step, status in view.plan_rows]
    values[2] = view.capability; values[16] = view.limitations; values[17] = view.final_report
    return normalize_render_values(values)


def stream_output(raw_state: dict, render_values: tuple[object, ...] | list[object],
                  button_update: object) -> tuple[object, ...]:
    """Build the one canonical output tuple used by every generator yield."""
    output = (raw_state, *normalize_render_values(render_values), button_update)
    if len(output) != RENDER_OUTPUT_COUNT + 2:
        raise ValueError("streaming output count does not match configured Gradio outputs")
    return output


def run_analysis_stream(mode: str, question: str, cancer_name: str, candidate: str,
                        ensembl_gene_id: str, research_mode: str):
    """Yield immediate and incremental UI updates from the backend callback."""

    if not _ACTIVE_RUN.acquire(blocking=False):
        values = list(EMPTY)
        values[0] = "## RUNNING\n\nAn analysis is already active. Duplicate execution was prevented."
        yield stream_output({"run_state": "duplicate_prevented"}, values,
                            gr.update(interactive=False, value="Analysis running…"))
        return
    state = {"fraction": 0.0, "message": "Starting", "history": [], "started": time.monotonic(),
             "run_id": _run_id(), "render_values": EMPTY, "checkpoint_count": 0,
             "completed_stage_ids": []}
    holder: dict[str, dict] = {}
    done = threading.Event()

    def update(fraction: float, *, desc: str = "", **_kwargs) -> None:
        previous = state["message"]
        if previous and previous != desc and previous != "Starting":
            state["history"].append(previous)
        state.update(fraction=float(fraction), message=desc or previous)

    def checkpoint_update(checkpoint: StageCheckpoint) -> None:
        state["render_values"] = apply_stage_checkpoint(state["render_values"], checkpoint)
        state["checkpoint_count"] += 1
        if checkpoint.status not in {"RUNNING", "FAILED"}:
            state["completed_stage_ids"].append(checkpoint.stage_id)

    def worker() -> None:
        try:
            holder["payload"] = run_selected_analysis(mode, question, cancer_name, candidate,
                                                       ensembl_gene_id, research_mode, progress=update,
                                                       requested_run_id=state["run_id"],
                                                       checkpoint_callback=checkpoint_update)
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    last = None
    try:
        while not done.is_set():
            fraction, message = state["fraction"], state["message"]
            elapsed = int(time.monotonic() - state["started"])
            signature = (fraction, message, elapsed, state["checkpoint_count"])
            if signature != last:
                match = re.match(r"Stage (\d+)/(\d+) — (.+)", message)
                current_index = int(match.group(1)) if match else 1
                total = int(match.group(2)) if match else len(WORKFLOW_STAGES)
                completed = max(0, current_index - 1)
                status = (f"## RUNNING\n\n**Run ID:** `{state['run_id']}` · **Current stage:** {message}  \n"
                          f"**Completed:** {completed} · **Skipped/Unavailable:** 0 · **Remaining:** {total - completed} · "
                          f"**Resolved:** {completed} / {total} stages · **Elapsed:** {elapsed // 60:02d}:{elapsed % 60:02d}")
                rows = [[item, "COMPLETED", "", "backend progress", "", "Completed", "Continue", "", ""] for item in state["history"]]
                rows.append([message, "RUNNING", "", "backend progress", "", "In progress", "Continue", "", ""])
                retained = list(state["render_values"])
                retained[0], retained[1] = status, rows
                yield stream_output({"run_state": "RUNNING", "progress": {"fraction": fraction, "current_stage": message,
                        "current_stage_index": current_index, "completed_steps": completed,
                        "resolved_steps": len(set(state["completed_stage_ids"])), "total_steps": total,
                        "elapsed_seconds": elapsed, "checkpointed_stages": state["completed_stage_ids"]}},
                       retained,
                       gr.update(interactive=False, value="Analysis running…"))
                last = signature
            time.sleep(.25)
        payload = holder["payload"]
        if "error" in payload and state["checkpoint_count"]:
            retained = list(state["render_values"])
            retained[0] = (f"## UI/WORKFLOW ERROR\n\n**Run ID:** `{state['run_id']}`  \n"
                           f"Checkpointed results were preserved. Error: {payload['error']}")
            retained[16] = (retained[16] or "") + "\n- A downstream error occurred; prior stage artifacts remain saved."
            final_render = retained
        else:
            final_render = render_selected(payload)
        yield stream_output(payload, final_render,
                            gr.update(interactive=True, value="Run biomedical analysis"))
    finally:
        _ACTIVE_RUN.release()


def filter_de_table(query: str, payload: dict | None) -> list[list[object]]:
    """Filter already-computed DE rows by gene/Ensembl text."""

    if not payload or payload.get("analysis_mode") != "REAL DATA" or payload.get("research_mode") != ResearchMode.GENOME_WIDE_DISCOVERY.value or "result" not in payload:
        return []
    view = build_genome_wide_run_view(GenomeWideAnalysisResult.model_validate(payload["result"]))
    needle = query.strip().casefold()
    return [list(row) for row in view.de_rows if not needle or needle in str(row[0]).casefold() or needle in str(row[1]).casefold()]


def load_persisted_run(run_id: str) -> tuple[object, ...]:
    """Restore a completed biomedical run without rerunning external analyses."""
    safe_run_id = run_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", safe_run_id):
        raise gr.Error("Enter a valid run ID.")
    result_path = load_settings().cache_dir / "runs" / safe_run_id / "complete_result.json"
    if not result_path.is_file():
        raise gr.Error(f"No completed saved run was found for {safe_run_id}.")
    result = GenomeWideAnalysisResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    payload = {"analysis_mode": "REAL DATA", "research_mode": result.research_mode.value,
               "result": result.model_dump(mode="json"),
               "run_metadata": {"run_id": safe_run_id, "agent": "OncoLncAI Supervisor",
                                "model": "restored persisted result", "elapsed_seconds": "restored"}}
    return (payload, *render_selected(payload))


def filter_literature_table(candidate: str, evidence_type: str, year: str,
                            relevance: str, payload: dict | None) -> list[list[object]]:
    """Filter structured literature rows without changing relevance decisions."""

    if not payload or payload.get("analysis_mode") != "REAL DATA" or "result" not in payload:
        return []
    if payload.get("research_mode") != ResearchMode.GENOME_WIDE_DISCOVERY.value:
        return []
    rows = build_genome_wide_run_view(GenomeWideAnalysisResult.model_validate(payload["result"])).literature_rows
    candidate_key, year_key = candidate.strip().casefold(), year.strip()
    return [list(row) for row in rows
            if (not candidate_key or candidate_key in str(row[0]).casefold())
            and (evidence_type == "All" or row[5] == evidence_type)
            and (not year_key or str(row[3]) == year_key)
            and (relevance == "All" or row[9] == relevance)]


def section_help(text: str) -> None:
    with gr.Accordion("How these results were obtained", open=False):
        gr.Markdown(text)


def create_app() -> gr.Blocks:
    with gr.Blocks(title="OncoLncAI — Agentic lncRNA Biomarker Discovery & Evidence Validation") as demo:
        gr.Markdown("# OncoLncAI — Agentic lncRNA Biomarker Discovery & Evidence Validation", elem_classes="hero")
        gr.Markdown("Integrates TCGA expression, survival analysis, literature evidence, external replication, and independent verification. Research use only; not clinical validation.")
        research_mode = gr.Radio(RESEARCH_MODE_CHOICES, value=ResearchMode.GENOME_WIDE_DISCOVERY.value, label="Research mode")
        question = gr.Textbox(value="Which lncRNAs are promising biomarkers for lung adenocarcinoma, and are they prognostically relevant?", label="Research question")
        cancer = gr.Dropdown(CANCER_CHOICES, value="Lung Adenocarcinoma (TCGA-LUAD)", label="TCGA cancer project", filterable=True)
        gr.Markdown("Registry validation status is available in provenance. Only TCGA-LUAD is currently LIVE_VALIDATED; other listed projects are available or experimental.")
        with gr.Column(visible=False, elem_classes="result-card") as candidate_section:
            gr.Markdown("### Candidate\nEnter a supported gene symbol or Ensembl ID. Known identifiers are resolved and conflicts prevent execution.")
            with gr.Row():
                candidate = gr.Textbox(value="", label="Gene symbol")
                ensembl = gr.Textbox(value="", label="Ensembl ID")
        run = gr.Button("Run biomedical analysis", variant="primary")
        with gr.Accordion("Advanced / Developer Options", open=False):
            mode = gr.Radio(["REAL DATA", "DEMO / SYNTHETIC"], value="REAL DATA", label="Analysis mode")
            gr.Markdown("Synthetic mode validates agent control flow for CI/offline development and never represents biomedical findings.")
            with gr.Row():
                restore_run_id = gr.Textbox(label="Restore completed run ID")
                restore = gr.Button("Load saved run")
        raw = gr.JSON(visible=False)
        overview = gr.Markdown(IDLE_STATUS_TEXT, elem_classes=["result-card", "run-status"])
        with gr.Tabs():
            with gr.Tab("Overview"):
                gr.Markdown("Resolved identifiers and backend sample counts appear in the status card above.")
            with gr.Tab("Agent Workflow"):
                section_help("Rows are emitted by the backend orchestration trace after validated tool inputs execute. Workflow-stage status is separate from low-level tool/observation events; replan reasons and provenance show why a branch changed.")
                workflow = gr.Dataframe(headers=["Goal", "Status", "Action", "Tool/function", "Validated input", "Output/observation", "Decision", "Replan reason", "Provenance"], interactive=False)
            with gr.Tab("Data & QC"):
                section_help("Tracks retrieved expression samples through primary-tumor/solid-normal selection, duplicate-aliquot removal, QC, annotation, and the final native-count DE cohort. Genes are all annotated features; lncRNAs are the retained GENCODE noncoding biotypes.")
                qc = gr.Markdown()
            with gr.Tab("Differential Expression"):
                section_help("Positive log2FC means higher tumor expression; negative means lower tumor expression. The p-value is nominal evidence and FDR is Benjamini–Hochberg adjusted. Qualification uses the displayed FDR, |log2FC|, and expression thresholds. The method/input/model shown below are the methods actually executed.")
                de_summary = gr.Markdown(); volcano = gr.HTML()
                de_filter = gr.Textbox(label="Search DE table", placeholder="Gene symbol or Ensembl ID")
                de_table = gr.Dataframe(headers=["Gene", "Ensembl ID", "Biotype", "Tumor expression", "Normal expression", "log2FC", "p-value", "FDR", "DE status"], interactive=False)
            with gr.Tab("Candidate Discovery"):
                section_help("Shows deterministic reduction from retrieved genes to annotated lncRNAs, DE-tested and DE-qualified lncRNAs, the bounded survival shortlist, and final ranked candidates. An LLM does not select candidates.")
                discovery = gr.Markdown()
            with gr.Tab("Survival"):
                section_help("HR > 1 associates higher expression with higher hazard; HR < 1 associates it with lower hazard. A 95% CI crossing 1 may not distinguish the association from HR=1. Log-rank compares median-defined groups; Cox models continuous expression; survival FDR corrects across candidates. Association is not causation.")
                survival = gr.Dataframe(headers=["Gene", "N", "Events", "HR", "95% CI", "p-value", "FDR", "Cutoff", "Status"], interactive=False)
            with gr.Tab("Adjusted Cox"):
                section_help("Adjusted Cox evaluates expression while accounting for available age, sex, and stage covariates. Inspect initial eligibility, complete cases, events, missingness, covariates, convergence, HR/CI, p-value and FDR. Adjusted association is neither clinical validation nor causality.")
                adjusted = gr.Dataframe(headers=["Gene", "Adjusted HR", "95% CI", "p-value", "FDR", "Initial N", "Complete-case N", "Events", "Missingness", "Covariates", "Convergence", "PH status"], interactive=False)
            with gr.Tab("PH Diagnostics"):
                section_help("The proportional-hazards assumption requires relative hazards to remain stable over time. PASS supports standard Cox interpretation; WARNING means not estimable/uncertain; FAIL indicates a detected violation and the claim must be downgraded.")
                ph = gr.Dataframe(headers=["Candidate", "Term", "Statistic", "p-value", "Decision"], interactive=False)
            with gr.Tab("Literature"):
                section_help("Retrieved articles are PubMed records returned by bounded candidate queries. Evidence-qualified articles require a separate structured evidence decision; retrieval alone is not support for expression, prognosis, or mechanism.")
                literature_summary = gr.Markdown("No analysis has been run yet.")
                with gr.Row():
                    literature_candidate = gr.Textbox(label="Filter candidate")
                    literature_evidence = gr.Dropdown(["All", "Retrieved article"], value="All", label="Evidence type")
                    literature_year = gr.Textbox(label="Publication year")
                    literature_relevance = gr.Dropdown(["All", "Not assessed"], value="All", label="Relevance")
                literature = gr.Dataframe(headers=["Candidate", "PMID", "Title", "Year", "Cancer", "Evidence type", "Expression", "Prognosis", "Mechanism", "Relevance", "Query", "Retrieved at"], interactive=False)
            with gr.Tab("GEO Validation"):
                section_help("Dataset discovery is not validation. A study must be compatible, a candidate must map to its platform, and deterministic expression or prognostic replication must succeed before external replication is claimed.")
                geo_summary = gr.Markdown(); geo = gr.Dataframe(headers=["GSE", "Platform", "Tumor N", "Normal N", "Mapped", "Unmapped", "Validation type", "Decision"], interactive=False)
            with gr.Tab("Candidate Ranking"):
                section_help("Evidence tiers combine the displayed DE, survival, adjusted-Cox/PH, literature and GEO components under the deterministic rule reported by the backend. No opaque AI confidence score is used; inspect each row and the selection rule for why a tier was assigned.")
                ranking = gr.Dataframe(headers=["Rank", "Gene", "Ensembl", "DE direction", "log2FC", "DE FDR", "HR", "Survival FDR", "Adjusted HR", "Cox FDR", "PH", "PMIDs", "GEO expression", "GEO prognosis", "Verifier", "Evidence tier", "Biotype"], interactive=False)
                gr.Markdown("Tiers are assigned by the deterministic rule shown in the Candidate Discovery section; they are evidence-prioritization labels, not biomarker probabilities.")
            with gr.Tab("Verifier"):
                section_help("ACCEPT retains a claim as supported; DOWNGRADE requires more cautious wording or tiering; REJECT removes an unsupported claim. Verifier decisions constrain the final report rather than changing computed statistics.")
                verifier = gr.Dataframe(headers=["Claim/check", "Decision", "Reason"], interactive=False)
            with gr.Tab("Limitations"):
                section_help("Limitations are generated from this run’s unavailable, ineligible, incomplete, failed, or assumption-violating branches. They define what the results do not establish.")
                limitations = gr.Markdown()
            with gr.Tab("Final Report"):
                section_help("The backend projects structured run, candidate, evidence, limitation, and conclusion records into compact tables. Every value is copied from deterministic results or explicitly marked Not evaluated; narrative synthesis cannot change them.")
                report = gr.Markdown()
            with gr.Tab("Provenance"):
                section_help("Provenance is copied from deterministic tool results and the run manifest: source dataset, project/accession, exact query where available, version/checksum, run ID, analysis module configuration, and retrieval time. It is never generated by an LLM.")
                provenance = gr.Dataframe(headers=["Source", "Project/accession", "Query", "Version"], interactive=False)
                manifest = gr.Code(language="json", label="Run manifest")
            with gr.Tab("Exports"):
                artifacts = gr.File(label="Backend-generated CSV/JSON artifacts", file_count="multiple")
        research_mode.change(candidate_input_state, research_mode, [candidate_section, candidate, ensembl])
        run.click(run_analysis_stream, [mode, question, cancer, candidate, ensembl, research_mode],
                  [raw, overview, workflow, qc, de_summary, volcano, de_table, discovery, survival,
                   adjusted, ph, literature_summary, literature, geo_summary, geo, ranking, verifier,
                   limitations, report, provenance, manifest, artifacts, run], api_name="run_analysis")
        restore.click(load_persisted_run, restore_run_id,
                      [raw, overview, workflow, qc, de_summary, volcano, de_table, discovery, survival,
                       adjusted, ph, literature_summary, literature, geo_summary, geo, ranking, verifier,
                       limitations, report, provenance, manifest, artifacts], api_name="load_saved_run")
        de_filter.change(filter_de_table, [de_filter, raw], de_table)
        for control in (literature_candidate, literature_evidence, literature_year, literature_relevance):
            control.change(filter_literature_table,
                           [literature_candidate, literature_evidence, literature_year, literature_relevance, raw], literature)
        with gr.Accordion("Runtime configuration (no secret values)", open=False):
            gr.JSON(value=runtime_configuration())
    return demo


demo = create_app()

if __name__ == "__main__":
    server_name = os.getenv("ONCOLNCAI_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("ONCOLNCAI_SERVER_PORT", "7860"))
    public_url = os.getenv("ONCOLNCAI_PUBLIC_URL", f"http://127.0.0.1:{server_port}")
    print(f"Open OncoLncAI in your browser: {public_url}", flush=True)
    demo.launch(server_name=server_name, server_port=server_port, show_error=True, css=APP_CSS)
