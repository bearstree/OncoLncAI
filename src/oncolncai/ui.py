"""Read-only presentation projections for structured research results."""

from __future__ import annotations

import html
import math

from pydantic import BaseModel, ConfigDict

from oncolncai.demo import OfflineDemoReport
from oncolncai.discovery_analysis import GenomeWideAnalysisResult
from oncolncai.real_analysis import RealAnalysisResult


class ResearchRunView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    overview: str
    plan_rows: tuple[tuple[str, str], ...]
    trace_rows: tuple[tuple[int, str, str], ...]
    capability: str
    replanning: str
    evidence: str
    computed_results: str
    verifier: str
    limitations: str
    final_report: str


class RealRunView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    header: str
    workflow_rows: tuple[tuple[str, str, str, str, str], ...]
    qc: str
    survival: str
    literature_rows: tuple[tuple[str, str, str], ...]
    ranking_rows: tuple[tuple[str, str, str, str, str], ...]
    verifier: str
    limitations: str
    report: str
    manifest: str


class GenomeWideRunView(BaseModel):
    """Complete UI projection; values remain those produced by the backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: str
    overview: str
    workflow_rows: tuple[tuple[str, ...], ...]
    qc: str
    de_summary: str
    volcano_html: str
    de_rows: tuple[tuple[object, ...], ...]
    discovery: str
    survival_rows: tuple[tuple[object, ...], ...]
    adjusted_rows: tuple[tuple[object, ...], ...]
    ph_rows: tuple[tuple[object, ...], ...]
    literature_summary: str
    literature_rows: tuple[tuple[object, ...], ...]
    geo_summary: str
    geo_rows: tuple[tuple[object, ...], ...]
    ranking_rows: tuple[tuple[object, ...], ...]
    verifier_rows: tuple[tuple[str, str, str], ...]
    limitations: str
    report: str
    provenance_rows: tuple[tuple[object, ...], ...]
    manifest: str
    artifacts: tuple[str, ...]
    consistency_warnings: tuple[str, ...] = ()


class GenomeWideRunSummary(BaseModel):
    """Canonical count semantics shared by every completed-run UI section."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    expression_samples_retrieved: int | None
    selected_expression_samples: int | None
    primary_tumor_samples_selected: int | None
    solid_tissue_normal_samples_selected: int | None
    samples_removed_during_qc: int | None
    de_tumor_n: int | None
    de_normal_n: int | None
    clinical_cases_retrieved: int | None
    survival_eligible_cases: int | None
    adjusted_cox_complete_cases: int | None
    genes_retrieved: int | None
    annotated_genes: int | None
    annotated_lncrnas: int | None
    expression_eligible_lncrnas: int | None
    de_tested_lncrnas: int | None
    de_qualified_lncrnas: int | None
    candidates_forwarded: int
    survival_screened: int
    survival_fdr_significant: int
    adjusted_cox_fdr_significant: int
    final_candidates: int
    literature_records: int
    candidates_with_literature: int
    geo_datasets_evaluated: int
    externally_replicated_candidates: int
    verifier_accept: int
    verifier_downgrade: int
    verifier_reject: int
    consistency_warnings: tuple[str, ...] = ()

def build_genome_wide_run_summary(result: GenomeWideAnalysisResult) -> GenomeWideRunSummary:
    """Derive labels/counts once from canonical structured backend fields."""

    prep = result.preparation.preparation if result.preparation else None
    de, discovery = result.differential_expression, result.discovery
    clinical = [getattr(item, "clinical_cases_retrieved", None) for item in result.survival_screen]
    clinical = [item for item in clinical if item is not None]
    complete_cases = {item.complete_case_count for item in result.adjusted_cox}
    warnings = []
    if de and result.manifest.sample_counts:
        if result.manifest.sample_counts.get("de_raw_tumor") != de.case_sample_count:
            warnings.append("manifest DE tumor count differs from DE result")
        if result.manifest.sample_counts.get("de_raw_normal") != de.control_sample_count:
            warnings.append("manifest DE normal count differs from DE result")
    candidate_ids = {item.gene_symbol for item in discovery.candidates} if discovery else set()
    ranking_ids = {item.gene_symbol for item in result.rankings}
    if not ranking_ids.issubset(candidate_ids):
        warnings.append("final ranking contains identifiers absent from candidate discovery")
    qualified_count = discovery.eligible_result_count if discovery else None
    forwarded_count = len(discovery.candidates) if discovery else 0
    if qualified_count is not None and forwarded_count > qualified_count:
        warnings.append("candidate count exceeds DE-qualified count")
    if len(result.survival_screen) > forwarded_count:
        warnings.append("survival-screened count exceeds candidates forwarded")
    if len(result.rankings) > forwarded_count:
        warnings.append("final ranking contains more candidates than candidate discovery")
    decisions = [item.decision.value for item in result.verification]
    literature = result.literature_summary
    return GenomeWideRunSummary(
        expression_samples_retrieved=prep.source_columns if prep else None,
        selected_expression_samples=prep.selected_samples if prep else None,
        primary_tumor_samples_selected=prep.tumor_samples if prep else None,
        solid_tissue_normal_samples_selected=prep.normal_samples if prep else None,
        samples_removed_during_qc=(prep.source_columns - prep.selected_samples) if prep else None,
        de_tumor_n=de.case_sample_count if de else None, de_normal_n=de.control_sample_count if de else None,
        clinical_cases_retrieved=max(clinical) if clinical else None,
        survival_eligible_cases=max((item.sample_count for item in result.survival_screen), default=None),
        adjusted_cox_complete_cases=(next(iter(complete_cases)) if len(complete_cases) == 1 else None),
        genes_retrieved=prep.genes_in_source if prep else None, annotated_genes=prep.annotated_genes if prep else None,
        annotated_lncrnas=prep.lncrnas_retained if prep else None,
        expression_eligible_lncrnas=de.selected_feature_count if de else None,
        de_tested_lncrnas=de.tested_feature_count if de else None,
        de_qualified_lncrnas=discovery.eligible_result_count if discovery else None,
        candidates_forwarded=forwarded_count,
        survival_screened=len(result.survival_screen),
        survival_fdr_significant=sum(item.fdr <= .05 for item in result.survival_screen),
        adjusted_cox_fdr_significant=sum(item.fdr is not None and item.fdr <= .05 for item in result.adjusted_cox),
        final_candidates=len(result.rankings), literature_records=literature.records_retrieved if literature else 0,
        candidates_with_literature=literature.candidates_with_records if literature else 0,
        geo_datasets_evaluated=len(result.manifest.geo_datasets_evaluated),
        externally_replicated_candidates=sum(item.geo_expression_replication == "REPLICATED" for item in result.rankings),
        verifier_accept=decisions.count("ACCEPT"), verifier_downgrade=decisions.count("DOWNGRADE"),
        verifier_reject=decisions.count("REJECT"), consistency_warnings=tuple(warnings),
    )


def _status(value: object) -> str:
    return str(getattr(value, "value", value)).upper().replace("_", " ")


def _n(value: float | None) -> str:
    return "Not evaluated" if value is None else f"{value:.4g}"


def _how(text: str) -> str:
    return f"\n<details><summary>How these results were obtained</summary>\n\n{text}\n\n</details>\n"


def _structured_final_report(result: GenomeWideAnalysisResult, summary: GenomeWideRunSummary) -> str:
    """Project deterministic backend records into compact, auditable report tables."""

    m, de = result.manifest, result.differential_expression
    source = f"{m.upstream_source} via {m.distribution_source}"
    lines = [
        "## Final Report", "", "### Run summary", "",
        "| Field | Value |", "|---|---|",
        f"| Research question | {m.research_question} |",
        f"| Cancer | {result.canonical_cancer_name} |", f"| TCGA project | {result.project_id} |",
        f"| Analysis / research mode | {getattr(result, 'analysis_mode', 'REAL')} / {result.research_mode.value} |",
        f"| Run status | {_status(result.status)} |", f"| Expression source | {source} |",
        "| Clinical source | TCGA PanCanAtlas |",
        f"| DE method | {de.configuration.method if de else 'Not evaluated'} |",
        f"| Tumor / normal N | {summary.de_tumor_n if summary.de_tumor_n is not None else 'Not evaluated'} / {summary.de_normal_n if summary.de_normal_n is not None else 'Not evaluated'} |",
        f"| Clinical / survival-eligible N | {summary.clinical_cases_retrieved if summary.clinical_cases_retrieved is not None else 'Not evaluated'} / {summary.survival_eligible_cases if summary.survival_eligible_cases is not None else 'Not evaluated'} |",
        f"| lncRNAs tested / final candidates | {summary.de_tested_lncrnas if summary.de_tested_lncrnas is not None else 'Not evaluated'} / {summary.final_candidates} |",
        _how(f"Project-specific expression and clinical sources were resolved from `{result.project_id}`. Counts and methods are copied from the run manifest and deterministic DE/survival outputs; run ID `{result.run_id}`, annotation `{m.gene_annotation_source_version}`."),
        "### Top candidates", "",
        "| Rank | Gene | Ensembl ID | Direction | log2FC | DE FDR | Univariate HR | Survival FDR | Adjusted HR | Adjusted Cox FDR | PH | Literature | GEO expression | GEO prognosis | Verifier | Evidence tier |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|",
    ]
    screens = {x.gene_symbol: x for x in result.survival_screen}
    adjusted = {x.gene: x for x in result.adjusted_cox}
    for rank, item in enumerate(result.rankings, 1):
        survival, cox = screens.get(item.gene_symbol), adjusted.get(item.gene_symbol)
        decision = next((v.decision.value for v in result.verification if item.gene_symbol.casefold() in v.check.casefold()), "ACCEPT")
        ph = "FAIL" if survival and survival.ph_violation else "PASS" if survival and survival.ph_violation is False else "Not evaluated"
        lines.append(f"| {rank} | {item.gene_symbol} | {item.ensembl_gene_id} | {'UP' if item.log2_fold_change > 0 else 'DOWN'} | {_n(item.log2_fold_change)} | {_n(item.de_fdr)} | {_n(survival.hazard_ratio if survival else None)} | {_n(item.survival_fdr)} | {_n(cox.adjusted_hazard_ratio if cox else None)} | {_n(item.adjusted_cox_fdr)} | {ph} | {len(item.literature_pmids)} PMID(s) | {item.geo_expression_replication} | Not evaluated | {decision} | {item.tier} |")
    if not result.rankings:
        lines.append("| — | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated |")
    rules = m.candidate_selection_rules
    lines += [_how(f"Native raw counts were tested with {de.configuration.method if de else 'the configured DE method'} and BH correction. Candidates passed FDR ≤ {rules.get('fdr_threshold', 'Not evaluated')}, |log2FC| ≥ {rules.get('absolute_log2_fc_threshold', 'Not evaluated')}, expression ≥ {rules.get('minimum_mean_cpm', 'Not evaluated')}, then deterministic top-N ranking and survival screening. Survival used patient-mapped expression, median KM groups, log-rank, continuous-expression Cox and BH across candidates. Adjusted Cox used complete cases and actual covariates recorded in its rows; PH diagnostics constrain interpretation. Missing branches remain Not evaluated."),
              "### Evidence summary", "", "| Evidence category | Result | Interpretation |", "|---|---|---|",
              f"| Differential expression | {summary.de_qualified_lncrnas if summary.de_qualified_lncrnas is not None else 'Not evaluated'} DE-qualified lncRNAs | Tumor/normal association under configured thresholds |",
              f"| Survival | {summary.survival_fdr_significant} FDR-significant candidates | Prognostic association, not causality |",
              f"| Adjusted Cox | {summary.adjusted_cox_fdr_significant} FDR-significant candidates | Independent association only where eligible and PH-compatible |",
              f"| Literature | {summary.literature_records} PubMed records linked to {summary.candidates_with_literature} candidates | Retrieval is not automatically claim support |",
              f"| External replication | {summary.externally_replicated_candidates} candidates replicated | Dataset discovery alone is not replication |",
              f"| Verifier | {summary.verifier_accept} ACCEPT / {summary.verifier_downgrade} DOWNGRADE / {summary.verifier_reject} REJECT | Final claims obey verifier decisions |",
              _how("Evidence components come from the DE, survival/Cox, PubMed, GEO and verifier artifacts for this run. PubMed searches were candidate- and cancer-specific and deduplicated by PMID. GEO required cancer/tissue, phenotype, platform and candidate compatibility plus a deterministic replication test and direction check."),
              "### Limitations", "", "| Analysis / component | Status | Limitation | Impact on interpretation |", "|---|---|---|---|" ]
    if result.limitations:
        for limitation in result.limitations:
            component = limitation.split(" ", 1)[0]
            status = next((word for word in ("NOT_AVAILABLE", "NOT_ELIGIBLE", "INTEGRATION_INCOMPLETE", "FAILED") if word in limitation), "LIMITED")
            lines.append(f"| {component} | {status} | {limitation.replace('|', '/')} | Claims from this component are unavailable or must be downgraded |")
    else:
        lines.append("| All recorded components | COMPLETED | None recorded | No additional run-specific limitation recorded |")
    lines += [_how("Limitations are emitted by unavailable, ineligible, failed, incomplete, or assumption-violating backend branches. They are preserved rather than converted to negative evidence."),
              "### Conclusions", "", "| Candidate | Supported claim | Evidence status | Verifier decision | Interpretation |", "|---|---|---|---|---|"]
    for item in result.rankings:
        survival = screens.get(item.gene_symbol)
        claims = ["Differentially expressed"]
        if survival and survival.fdr <= .05:
            claims.append("Associated with survival")
        if item.adjusted_cox_fdr is not None and item.adjusted_cox_fdr <= .05:
            claims.append("Adjusted prognostic association")
        if item.literature_pmids:
            claims.append("Literature retrieved")
        claims.append("Externally replicated" if item.geo_expression_replication == "REPLICATED" else "Not externally replicated")
        decision = next((v.decision.value for v in result.verification if item.gene_symbol.casefold() in v.check.casefold()), "ACCEPT")
        lines.append(f"| {item.gene_symbol} | {'; '.join(claims)} | {item.tier} | {decision} | Research-prioritization evidence; not a clinically validated biomarker |")
    lines += [_how("Conclusion claims are assembled only from structured component results and then constrained by deterministic verifier decisions. They do not convert association into causality or evidence tiers into clinical validation."),
              "### Narrative synthesis", "", result.final_report]
    return "\n".join(lines)


def _volcano_svg(result: GenomeWideAnalysisResult) -> str:
    """Visualize backend FC/FDR and backend-selected candidate identities."""

    de = result.differential_expression
    if de is None or not de.results:
        return "<p>Volcano plot unavailable because DE was not completed.</p>"
    max_x = max(1.0, max(abs(x.log2_fold_change) for x in de.results))
    y_values = [-math.log10(max(x.adjusted_p_value, 1e-300)) for x in de.results]
    max_y = max(1.0, max(y_values))
    qualified = set(result.discovery.qualified_feature_ids) if result.discovery else set()
    candidates = {item.ensembl_gene_id: item for item in result.rankings}
    circles, labels = [], []
    for row, y_value in zip(de.results, y_values):
        x = 400 + row.log2_fold_change / max_x * 360
        y = 360 - y_value / max_y * 320
        label = html.escape(row.gene_symbol or row.feature_id)
        candidate = candidates.get(row.feature_id)
        point_status = "shortlisted candidate" if candidate else "DE-qualified" if row.feature_id in qualified else "tested"
        color, radius = ("#6d28d9", 5) if candidate else (("#d97706", 3) if row.feature_id in qualified else ("#83939a", 2.2))
        tier = candidate.tier if candidate else "not ranked"
        circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"><title>{label} | {row.feature_id} | log2FC={row.log2_fold_change:.3g} | p={row.p_value:.3g} | FDR={row.adjusted_p_value:.3g} | {point_status} | {tier}</title></circle>')
        if candidate:
            labels.append(f'<text x="{x + 5:.1f}" y="{max(14, y - 5):.1f}" font-size="10">{label}</text>')
    rules = result.manifest.candidate_selection_rules
    fc, alpha = float(rules.get("absolute_log2_fc_threshold", 0)), float(rules.get("fdr_threshold", .05))
    left, right = 400 - fc / max_x * 360, 400 + fc / max_x * 360
    fdr_y = 360 - (-math.log10(alpha) / max_y) * 320
    thresholds = (f'<line x1="{left:.1f}" y1="30" x2="{left:.1f}" y2="360" stroke="#d97706" stroke-dasharray="5 4"/>'
                  f'<line x1="{right:.1f}" y1="30" x2="{right:.1f}" y2="360" stroke="#d97706" stroke-dasharray="5 4"/>'
                  f'<line x1="40" y1="{fdr_y:.1f}" x2="770" y2="{fdr_y:.1f}" stroke="#d97706" stroke-dasharray="5 4"/>')
    return ('<div class="volcano"><svg viewBox="0 0 800 400" role="img" aria-label="Volcano plot">'
            '<line x1="40" y1="360" x2="770" y2="360" stroke="#64748b"/><line x1="400" y1="30" x2="400" y2="360" stroke="#cbd5e1"/>'
            + thresholds + "".join(circles) + "".join(labels)
            + '<text x="350" y="392">log2 fold change</text><text x="8" y="25">−log10(FDR)</text></svg>'
              '<p><span style="color:#6d28d9">● Shortlisted</span> · <span style="color:#d97706">● DE-qualified</span> · ● Tested</p></div>')


def _scientific_overview(result: GenomeWideAnalysisResult, summary: GenomeWideRunSummary) -> str:
    m = result.manifest
    performed = ["resolved the cancer and inspected TCGA capabilities"]
    if result.preparation:
        performed.append("retrieved and quality-checked genome-wide expression with lncRNA annotation")
    if result.differential_expression:
        performed.append("compared primary tumors with solid-tissue normal samples")
    if result.survival_screen:
        performed.append("screened deterministic DE candidates for survival association")
    if result.adjusted_cox:
        performed.append("fit adjusted Cox models and evaluated PH diagnostics")
    if result.literature_summary:
        performed.append("retrieved candidate-linked PubMed records")
    if any(item.analysis == "geo_expression_replication" for item in result.capabilities):
        performed.append("attempted GEO replication")
    if result.verification:
        performed.append("verified final claims")
    warning = ("\n\n> **RESULT CONSISTENCY WARNING:** " + "; ".join(summary.consistency_warnings)) if summary.consistency_warnings else ""
    return (
        f"### {_status(result.status)}\n\n**Research question:** {m.research_question}  \n"
        f"**Resolved cancer:** {result.canonical_cancer_name} · **TCGA project:** `{result.project_id}`  \n"
        f"**Mode:** REAL DATA · **Research mode:** Genome-Wide Discovery · **Run ID:** `{result.run_id}`  \n"
        f"**Manifest time:** {m.timestamp.isoformat()}\n\n#### What this analysis did\n"
        + ". It ".join(performed).capitalize() + ".\n\n#### Key results\n"
        f"**COHORT** — Expression retrieved: {summary.expression_samples_retrieved}; selected tumors: {summary.primary_tumor_samples_selected}; "
        f"selected normals: {summary.solid_tissue_normal_samples_selected}; DE tumor N: {summary.de_tumor_n}; DE normal N: {summary.de_normal_n}; "
        f"clinical cases retrieved: {summary.clinical_cases_retrieved}; survival-eligible: {summary.survival_eligible_cases}; Cox complete cases: {summary.adjusted_cox_complete_cases}  \n"
        f"**GENOME-WIDE ANALYSIS** — Genes retrieved: {summary.genes_retrieved}; annotated lncRNAs: {summary.annotated_lncrnas}; "
        f"expression-eligible: {summary.expression_eligible_lncrnas}; DE-tested: {summary.de_tested_lncrnas}; DE-qualified: {summary.de_qualified_lncrnas}  \n"
        f"**CANDIDATES** — Forwarded: {summary.candidates_forwarded}; survival-screened: {summary.survival_screened}; "
        f"survival FDR-significant: {summary.survival_fdr_significant}; adjusted-Cox FDR-significant: {summary.adjusted_cox_fdr_significant}; final: {summary.final_candidates}  \n"
        f"**EVIDENCE** — PubMed records: {summary.literature_records}; candidates with literature: {summary.candidates_with_literature}; "
        f"GEO datasets evaluated: {summary.geo_datasets_evaluated}; externally replicated: {summary.externally_replicated_candidates}; "
        f"verifier ACCEPT/DOWNGRADE/REJECT: {summary.verifier_accept}/{summary.verifier_downgrade}/{summary.verifier_reject}"
        + warning
    )


def _canonical_qc(result: GenomeWideAnalysisResult, summary: GenomeWideRunSummary) -> str:
    if not result.preparation:
        return "Data preparation was not completed."
    prep, m = result.preparation.preparation, result.manifest
    return (
        f"**Expression:** {m.upstream_source} via {m.distribution_source} · **Clinical/survival:** TCGA PanCanAtlas  \n"
        f"**Expression samples retrieved:** {summary.expression_samples_retrieved} → **selected:** {summary.selected_expression_samples} → "
        f"**primary tumors:** {summary.primary_tumor_samples_selected} / **solid normals:** {summary.solid_tissue_normal_samples_selected} → "
        f"**native-count DE cohort:** {summary.de_tumor_n} tumors / {summary.de_normal_n} normals  \n"
        f"**Samples removed during selection/QC:** {summary.samples_removed_during_qc} · **Duplicate aliquots removed:** {prep.duplicate_aliquots_removed} · "
        f"**Duplicate gene rows removed:** {prep.duplicate_gene_rows_removed}  \n"
        f"**Genes retrieved:** {summary.genes_retrieved} · **annotated genes:** {summary.annotated_genes} · **annotated lncRNAs:** {summary.annotated_lncrnas} · "
        f"**expression-eligible lncRNAs:** {summary.expression_eligible_lncrnas} · **DE-tested lncRNAs:** {summary.de_tested_lncrnas} · **DE-qualified:** {summary.de_qualified_lncrnas}  \n"
        f"**Clinical cases retrieved:** {summary.clinical_cases_retrieved} · **Survival-eligible cases:** {summary.survival_eligible_cases} · "
        f"**Adjusted-Cox complete cases:** {summary.adjusted_cox_complete_cases}  \n"
        f"**Annotation:** {m.gene_annotation_source_version} · **Preprocessing:** {m.expression_preprocessing} · "
        f"**DE representation:** {m.de_expression_source.get('representation', m.expression_data_type)}"
    )


def build_genome_wide_run_view(result: GenomeWideAnalysisResult) -> GenomeWideRunView:
    """Format a genome-wide run without changing calculations or conclusions."""

    m = result.manifest
    prep = result.preparation.preparation if result.preparation else None
    de, discovery = result.differential_expression, result.discovery
    summary = build_genome_wide_run_summary(result)
    overview = (
        f"### {_status(result.status)}\n\n**Run ID:** `{result.run_id}` · **Mode:** REAL DATA · **Research mode:** `{result.research_mode.value}`  \n"
        f"**Cancer:** {result.canonical_cancer_name} · **TCGA project:** `{result.project_id}`  \n"
        f"**Start/manifest time:** {m.timestamp.isoformat()} · **Current stage:** Final synthesis · **Final status:** {_status(result.status)}  \n\n"
        "| Tumor | Normal | Clinical | Survival screened | Genes retrieved | lncRNAs analyzed | DE-qualified | Final candidates | GEO evaluated |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| {de.case_sample_count if de else 'NA'} | {de.control_sample_count if de else 'NA'} | {len(result.preparation.clinical) if result.preparation else 'NA'} | "
        f"{len(result.survival_screen)} | {prep.genes_in_source if prep else 'NA'} | {de.tested_feature_count if de else 'NA'} | "
        f"{discovery.eligible_result_count if discovery else 'NA'} | {len(result.rankings)} | {len(m.geo_datasets_evaluated)} |"
    )
    overview = _scientific_overview(result, summary)
    workflow = tuple((e.goal, _status(e.execution_status), e.planned_action, e.selected_tool,
                      e.validated_arguments, e.observation, e.decision, e.replan_reason or "", "See Provenance")
                     for e in result.events)
    qc = "Data preparation was not completed."
    if prep:
        qc = (f"**Expression:** {m.upstream_source} via {m.distribution_source} · **Clinical:** TCGA/GDC  \n"
              f"**Tumor N:** {prep.tumor_samples} · **Normal N:** {prep.normal_samples} · **Genes before filtering:** {prep.genes_in_source} · "
              f"**After annotation:** {prep.annotated_genes} · **lncRNAs retained:** {prep.lncrnas_retained}  \n"
              f"**Removed samples:** {prep.source_columns - prep.selected_samples} · **Duplicate aliquots removed:** {prep.duplicate_aliquots_removed} · "
              f"**Duplicate gene rows removed:** {prep.duplicate_gene_rows_removed}  \n"
              f"**Annotation:** {m.gene_annotation_source_version} · **Selection:** primary tumor/solid-tissue normal, one aliquot per participant  \n"
              f"**Preprocessing:** {m.expression_preprocessing} · **DE source:** {m.de_expression_source.get('representation', m.expression_data_type)}")
    qc = _canonical_qc(result, summary)
    de_summary = "**Status:** NOT ELIGIBLE or INTEGRATION INCOMPLETE. See workflow and limitations."
    de_rows = ()
    if de:
        c = de.configuration
        de_summary = (f"**DE tumor N:** {summary.de_tumor_n} · **DE normal N:** {summary.de_normal_n} · "
                      f"**expression-eligible lncRNAs:** {summary.expression_eligible_lncrnas} · **lncRNAs tested:** {summary.de_tested_lncrnas}  \n"
                      f"**Input:** {c.input_data_type} ({c.expression_transform}) · **Normalization/model:** {c.normalization} / {c.method} · **Test:** {c.test_method}  \n"
                      f"**FDR:** {c.multiple_testing_method}, α={c.alpha} · **|log2FC| threshold:** {m.candidate_selection_rules.get('absolute_log2_fc_threshold', 'NA')}")
        de_rows = tuple((x.gene_symbol or "", x.feature_id, "lncRNA", _n(x.case_mean_cpm), _n(x.control_mean_cpm),
                         _n(x.log2_fold_change), _n(x.p_value), _n(x.adjusted_p_value), "PASS" if x.significant else "NOT SIGNIFICANT")
                        for x in de.results)
    rules = m.candidate_selection_rules
    discovery_text = (f"**Genes retrieved:** {summary.genes_retrieved} → **annotated lncRNAs:** {summary.annotated_lncrnas} → "
                      f"**DE-tested lncRNAs:** {summary.de_tested_lncrnas} → **DE-qualified:** {summary.de_qualified_lncrnas} → "
                      f"**forwarded to survival:** {summary.candidates_forwarded} → **final candidates:** {summary.final_candidates}  \n"
                      f"Rules: FDR ≤ {rules.get('fdr_threshold', 'NA')}; |log2FC| ≥ {rules.get('absolute_log2_fc_threshold', 'NA')}; "
                      f"mean expression ≥ {rules.get('minimum_mean_cpm', 'NA')}; top N={rules.get('top_n', 'NA')}. Deterministic selection; not an LLM pick.")
    survival = tuple((x.gene_symbol, x.sample_count, x.event_count, _n(x.hazard_ratio),
                      f"{_n(x.confidence_interval_lower)}–{_n(x.confidence_interval_upper)}", _n(x.p_value), _n(x.fdr), "median", "COMPLETED")
                     for x in result.survival_screen)
    adjusted = tuple((x.gene, _n(x.adjusted_hazard_ratio), f"{_n(x.confidence_interval_lower)}–{_n(x.confidence_interval_upper)}",
                      _n(x.p_value), _n(x.fdr), x.initial_survival_eligible_count, x.complete_case_count, x.event_count,
                      str(x.missingness_by_covariate), ", ".join(x.covariates_used),
                      x.convergence_status.value, "FAIL" if x.overall_ph_violation else "PASS" if x.overall_ph_violation is False else "WARNING")
                     for x in result.adjusted_cox)
    ph = tuple((x.gene, d.term, _n(d.statistic), _n(d.p_value), "FAIL" if d.violation else "PASS" if d.violation is False else "WARNING")
               for x in result.adjusted_cox for d in x.diagnostics)
    literature = tuple((gene, p.pmid, p.title, p.publication_year or "Unavailable", result.canonical_cancer_name,
                        "Retrieved article", "Not extracted", "Not extracted", "Not extracted", "Not assessed",
                        p.query, p.retrieval_timestamp.isoformat())
                       for gene, papers in result.literature.items() for p in papers)
    ls = result.literature_summary
    literature_summary = (
        "**Literature evidence collected from NCBI PubMed**  \n"
        f"Retrieved: **{ls.records_retrieved if ls else 0}** · Unique: **{ls.unique_records if ls else 0}** · "
        f"Candidate-linked: **{ls.candidate_linked_records if ls else 0}** · "
        f"Candidates supported: **{ls.candidates_with_records if ls else 0} / {ls.candidates_searched if ls else 0}**  \n"
        f"Evidence-qualified: **{ls.evidence_qualified_records if ls and ls.evidence_qualified_records is not None else 'Not assessed'}** · "
        "Expression/prognostic/mechanistic subtype counts: **Not assessed by this retrieval stage**"
    )
    geo_cap = next((c for c in result.capabilities if c.analysis == "geo_expression_replication"), None)
    geo_summary = f"**{_status(geo_cap.status) if geo_cap else 'NOT AVAILABLE'}** — {geo_cap.reason if geo_cap and geo_cap.reason else 'No compatible dataset selected.'}"
    geo = tuple((x.accession, x.platform, x.tumor_count, x.normal_count, ", ".join(x.mapped_candidates),
                 ", ".join(x.unmapped_candidates), "EXPRESSION REPLICATION", "COMPLETED") for x in result.geo_replications)
    replicated = {x.accession for x in result.geo_replications}
    geo += tuple((accession, "Not selected", "Unknown", "Unknown", "Not evaluated", "Not evaluated",
                  "NOT EVALUATED", _status(geo_cap.status) if geo_cap else "NOT AVAILABLE")
                 for accession in m.geo_datasets_evaluated if accession not in replicated)
    screens, cox = {x.gene_symbol: x for x in result.survival_screen}, {x.gene: x for x in result.adjusted_cox}
    candidates = {x.gene_symbol: x for x in discovery.candidates} if discovery else {}
    ranking = []
    for rank, x in enumerate(result.rankings, 1):
        s, a, candidate = screens.get(x.gene_symbol), cox.get(x.gene_symbol), candidates.get(x.gene_symbol)
        decision = next((v.decision.value for v in result.verification if x.gene_symbol.casefold() in v.check.casefold()), "ACCEPT")
        ranking.append((rank, x.gene_symbol, x.ensembl_gene_id, "UP" if x.log2_fold_change > 0 else "DOWN", _n(x.log2_fold_change), _n(x.de_fdr),
                        _n(s.hazard_ratio if s else None), _n(x.survival_fdr), _n(a.adjusted_hazard_ratio if a else None), _n(x.adjusted_cox_fdr),
                        "FAIL" if s and s.ph_violation else "PASS" if s and s.ph_violation is False else "NOT EVALUATED",
                        len(x.literature_pmids), x.geo_expression_replication, "NOT EVALUATED", decision, x.tier,
                        candidate.gene_biotype if candidate else "lncRNA"))
    return GenomeWideRunView(
        status=_status(result.status), overview=overview, workflow_rows=workflow, qc=qc, de_summary=de_summary,
        volcano_html=_volcano_svg(result), de_rows=de_rows, discovery=discovery_text, survival_rows=survival,
        adjusted_rows=adjusted, ph_rows=ph, literature_summary=literature_summary,
        literature_rows=literature, geo_summary=geo_summary, geo_rows=geo,
        ranking_rows=tuple(ranking), verifier_rows=tuple((x.check, x.decision.value, x.detail) for x in result.verification),
        limitations="\n".join(f"- {x}" for x in result.limitations) or "- None recorded",
        report=_structured_final_report(result, summary),
        provenance_rows=tuple((p.source, p.source_id or "", p.query or "", p.dataset_version or "") for p in m.provenance),
        manifest=m.model_dump_json(indent=2), artifacts=m.output_artifacts,
        consistency_warnings=summary.consistency_warnings,
    )


def build_real_run_view(result: RealAnalysisResult) -> RealRunView:
    c = result.capabilities
    qc = "Capability retrieval unavailable." if c is None else (f"**Project:** `{c.project_id}` · **RNA-seq:** {c.rna_seq_available} · **Clinical:** {c.clinical_available}  \n"
                                                                  f"**Tumors:** {c.tumor_samples} · **Normals:** {c.normal_samples} · **Survival-eligible:** {c.survival_samples}")
    survival = "Survival unavailable."
    if result.survival:
        x = result.survival
        survival = (f"`{x.gene}` · N={x.sample_count} · events={x.event_count} · HR={x.hazard_ratio:.3g} · 95% CI {x.confidence_interval_lower:.3g}–{x.confidence_interval_upper:.3g} · "
                    f"Cox p={x.cox_p_value:.3g} · log-rank p={x.log_rank_p_value:.3g} · median cutoff={x.configuration.expression_cutoff:.3g}")
    v = result.verification
    verifier = "Verifier unavailable." if v is None else f"**{'ACCEPT' if v.passed else 'REJECT'}** — mismatches={len(v.numeric_mismatches)}, missing sources={len(v.missing_sources)}, unsupported={len(v.unsupported_claims)}"
    return RealRunView(
        header=f"## {_status(result.status)}\n\n**Run ID:** `{result.run_id}` · **REAL DATA** · `SINGLE_CANDIDATE_VALIDATION`  \n**Cancer:** {result.canonical_cancer_name} · **Project:** `{result.project_id}` · **Gene:** `{result.candidate}`",
        workflow_rows=tuple((x.label, _status(x.status), x.tool, x.observation, x.decision) for x in result.workflow), qc=qc, survival=survival,
        literature_rows=tuple((x.pmid, str(x.publication_year or "Unavailable"), x.title) for x in result.papers),
        ranking_rows=((result.ranking.candidate, result.ranking.evidence_tier, str(result.ranking.survival_significant), str(result.ranking.ph_assumption_met), result.ranking.external_validation),),
        verifier=verifier, limitations="\n".join(f"- {x}" for x in result.limitations) or "- None recorded",
        report=result.final_report, manifest=result.manifest.model_dump_json(indent=2))


def build_demo_run_view(question: str, report: OfflineDemoReport) -> ResearchRunView:
    completed, skipped = set(report.completed_steps), set(report.skipped_steps)
    rows = tuple((x, "completed" if x in completed else "skipped" if x in skipped else "pending") for x in report.initial_steps)
    return ResearchRunView(
        overview=f"**Question:** {question}\n\n**Resolved cancer:** {report.canonical_cancer_name}  \n**TCGA project:** `{report.tcga_project_id}`  \n**Run mode:** `{report.mode}`",
        plan_rows=rows, trace_rows=tuple((i, x.replace("_", " ").title(), "Recorded by agent loop") for i, x in enumerate(report.event_types, 1)),
        capability="Synthetic fixture: 30 tumors, 2 normals and survival available.",
        replanning="### Observation-driven replanning\n\nDE was skipped after the synthetic insufficient-normal observation.",
        evidence="**Not evaluated in DEMO / SYNTHETIC mode.**", computed_results="**Not evaluated in DEMO / SYNTHETIC mode.**",
        verifier="Control-flow validation: **PASS**. Scientific claims: **Not evaluated**.",
        limitations="\n".join(f"- {x}" for x in report.warnings) + f"\n\n{report.disclaimer}",
        final_report="Synthetic control-flow validation completed. **No biomedical conclusion was produced.**")
