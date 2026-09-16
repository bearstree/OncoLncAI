---
title: OncoLncAI
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.23.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
---

# OncoLncAI

## Agentic lncRNA Biomarker Discovery and Evidence Validation

OncoLncAI is a research-oriented agent that integrates TCGA expression and
clinical data, differential-expression analysis, survival modeling, literature
evidence, external replication, and claim verification to identify and assess
computational lncRNA biomarker candidates. It does not establish clinical
validity or utility.

The application supports genome-wide discovery without a supplied gene and
targeted evaluation of a known lncRNA. Analyses run in stages, preserve source
and method provenance, expose results as they complete, and continue with
scientifically defensible branches when a nonfatal analysis is unavailable.

## Research Modes

| Mode | Starting point | Purpose |
|---|---|---|
| **Genome-Wide Discovery** | Cancer name or TCGA project | Discovers lncRNA candidates from available expression data, then screens eligible candidates using clinical, survival, literature, and external evidence. No candidate gene is required. |
| **Targeted Candidate Validation** | Cancer plus lncRNA/gene identifier | Evaluates available expression, survival, literature, and external-validation evidence for a specified candidate. |

An explicitly labeled synthetic/demo path exists for development and deployment
checks. A requested REAL analysis never silently falls back to synthetic
biomedical results.

## Agent Workflow

The supervisor converts the research question into executable tasks and selects
registered, schema-validated tools. Tool arguments are checked before execution;
the language model does not directly produce biomedical statistics.

After each tool runs, the agent records a structured observation and updates the
remaining plan. Replanning can follow insufficient normal samples, unavailable
clinical covariates, Cox-model ineligibility, candidate mapping failure, ambiguous
GEO phenotypes, or an unavailable external service. A failed optional branch does
not erase completed evidence or force the entire run into an unexplained stopped
state.

```text
Research Question
      ↓
Planner / Replanner
      ↓
Validated Tool Selection
      ↓
Biomedical + Statistical Tools
      ↓
Observations / Stage Checkpoints
      ↓
Evidence Integration
      ↓
Independent Verifier
      ↓
Grounded Final Report
```

Stages maintain explicit states such as pending, running, completed, not
eligible, not available, integration incomplete, skipped after replan, and
failed. The UI reports these observable states and tool activities without
exposing private chain-of-thought.

### LLM and deterministic-code responsibilities

| LLM-assisted reasoning | Deterministic Python |
|---|---|
| Research-question interpretation | Sample counting and identifier mapping |
| Planning and validated tool selection | Expression retrieval, preprocessing, and QC |
| Observation interpretation and replanning | Differential expression, p-values, and BH/FDR |
| Semantic evidence interpretation where rules are insufficient | Kaplan–Meier, log-rank, Cox regression, confidence intervals, and PH diagnostics |
| Explanation of already verified evidence | PubMed record handling, GEO replication statistics, filtering, checkpointing, and provenance |

The LLM cannot invent sample counts, log2 fold changes, p-values, FDR values,
hazard ratios, confidence intervals, PMIDs, or GEO accessions. Final synthesis may
explain verified results but may not alter them.

## Biomedical Analysis Path

A genome-wide run can follow this path, subject to project capability and data
eligibility:

```text
Cancer / TCGA project resolution
→ expression and clinical-data acquisition
→ data QC and lncRNA annotation
→ differential expression and candidate discovery
→ survival screening with multiple-testing correction
→ adjusted Cox where eligible
→ proportional-hazards diagnostics
→ candidate-specific PubMed evidence
→ compatible GEO replication attempt
→ deterministic evidence tiering
→ verification
→ final synthesis
```

Not every stage executes for every project or candidate. The run records why an
analysis was completed, skipped, unavailable, or ineligible.

## Data Sources

- **TCGA / NCI Genomic Data Commons (GDC)** are the upstream sources for cancer,
  biospecimen, clinical, and sequencing data.
- **UCSC Xena** distributes GDC-harmonized TCGA expression and phenotype datasets
  used by the genome-wide retrieval path. The application records the actual Xena
  dataset URL and identifier rather than implying that each underlying GDC file
  was downloaded individually.
- **GDC raw STAR counts** are available through the native raw-count client for
  count-based differential-expression paths where applicable.
- **PubMed / NCBI E-utilities** provide citation metadata and abstracts. PMIDs
  remain attached to extracted evidence.
- **NCBI GEO** supplies candidate external cohorts, Series Matrix data, and GPL
  annotations when compatible material is available.

External records retain dataset identifiers, accessions, retrieval information,
and relevant configuration in run artifacts.

## Statistical and Evidence Methods

### Differential expression

`differential_expression.py` is the canonical analysis interface. Raw integer
counts use the count-based backend; transformed or continuous expression is not
passed into a negative-binomial model. The deterministic path applies
low-expression filtering and calculates log2 fold changes, p-values, and
Benjamini–Hochberg FDR values. Candidate selection uses explicit, recorded
thresholds rather than model-generated preferences.

### Survival analysis

The survival workflow provides Kaplan–Meier summaries, log-rank testing,
univariate Cox proportional-hazards models, and multiple-testing correction.
Eligible candidates proceed to complete-case adjusted Cox models using available
covariates such as age, sex, and stage. Schoenfeld-residual time-trend diagnostics
evaluate the proportional-hazards assumption and constrain interpretation when it
fails or cannot be estimated.

A prognostic association is not causality. An adjusted prognostic association is
not clinical validation.

### Literature evidence

Candidate-specific PubMed retrieval records and deduplicates real citations,
links evidence to the candidate, and distinguishes expression, prognostic, and
mechanistic evidence. A retrieved article is not automatically counted as
supporting biomarker evidence; relevance and claim support are evaluated before
synthesis.

### GEO external validation

GEO discovery evaluates cancer/disease match, tissue, assay/platform,
case/control or survival phenotype availability, sample size, and annotation
compatibility. Where supported, the integration retrieves standardized Series
Matrix data, maps GPL annotations and candidates, and performs deterministic
tumor-versus-normal expression replication. GEO prognostic replication is shown
as not evaluated by the current production workflow rather than being inferred
from incomplete survival metadata.

“Dataset discovered” and “external replication established” are separate
outcomes. Ambiguous phenotypes, incompatible annotations, or unsupported
supplementary formats produce an explicit not-eligible or integration-incomplete
result. Arbitrary supplementary formats may require a dedicated adapter.

## Evidence Ranking, Verification, and Provenance

Genome-wide candidates first satisfy explicit DE filters. Tier A then requires
direction-consistent GEO expression replication, significant adjusted-Cox FDR,
and no PH violation; Tier B requires significant survival FDR and no PH
violation; remaining ranked candidates are Tier C. The targeted mode uses its
documented Tier B/C/D rules based on available survival, PH, and literature
evidence. The implementation does not present an unexplained LLM confidence
score, and missing evidence remains missing rather than being treated as negative.

An independent deterministic-first verifier checks numerical consistency,
identifiers, sources, provenance, prohibited causal wording, and correspondence
between claims and computed evidence. Semantic model review is reserved for
claim-support questions that deterministic rules cannot resolve. Outcomes are:

- **ACCEPT** — the claim is supported as written;
- **DOWNGRADE** — evidence requires more cautious wording or tiering;
- **REJECT** — the claim is unsupported and is excluded from the conclusion.

Run metadata records data sources, methods, parameters, software/source version,
sample information, stage states, and output artifacts. This makes each reported
claim traceable to its execution evidence.

## Progressive Checkpointing

Results are saved stage by stage: QC summaries after data checks, DE tables and
plots after differential expression, survival/Cox/PH outputs after modeling, and
literature, GEO, ranking, verifier, and report artifacts as those stages finish.
Completed results remain available when a later optional branch fails. Saved runs
can be restored without repeating completed biomedical analysis, subject to the
configured storage lifetime.

## Web Application

The Gradio application provides:

- TCGA cancer selection and both research modes;
- live overall progress, current tool activity, and per-stage status;
- Data & QC summaries;
- differential-expression results, table, and volcano plot;
- candidate discovery and survival, adjusted-Cox, and PH results;
- PubMed evidence counts and GEO validation status;
- candidate ranking with evidence tiers;
- verifier outcomes and a structured final report;
- downloadable artifacts and saved-run recovery.

The root Hugging Face `app.py` is a thin launcher over the same production
`src/oncolncai` implementation; it does not duplicate the biomedical pipeline.

## Run Locally

Python 3.12 is the validated release target.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --requirement requirements.lock
python app.py
```

Open `http://127.0.0.1:7860`. Configuration is environment-based; see
`.env.example`. Ollama defaults to `qwen2.5-coder:14b`; Mock and OpenAI-compatible
providers use the same provider interface.

Command-line REAL analysis:

```bash
python examples/run_real_analysis.py \
  --research-mode genome-wide \
  --cancer "lung adenocarcinoma" \
  --summary
```

## Docker

```bash
docker build --tag oncolncai:0.1.0-public .
docker run --rm --publish 127.0.0.1:7860:7860 \
  --env-file .env oncolncai:0.1.0-public
```

Or with Compose:

```bash
docker compose up --build
```

The container binds Gradio to `0.0.0.0`; use `127.0.0.1` from the host browser.
For host Ollama, configure an address reachable from the container, commonly
`http://host.docker.internal:11434`.

## Hugging Face Space

The public Gradio Space is
[weiyi/OncoLncAI](https://huggingface.co/spaces/weiyi/OncoLncAI). It uses the
same production implementation as the canonical GitHub source. Hosted CPU,
memory, network, request-time, and ephemeral-storage limits may prevent an
uncached genome-wide REAL run; the application reports the limitation rather
than substituting synthetic findings.

## Version and Reproducibility

The prepared public version is **0.1.0**. The canonical repository is
[bearstree/OncoLncAI](https://github.com/bearstree/OncoLncAI). The `v0.1.0` tag
is created only after both GitHub and the Hugging Face Space pass remote
verification.

Reproducibility is supported by versioned source and locked dependencies,
`deployment/deployment_manifest.json`, the sorted public file inventory and
package hash, and run-specific manifests/checkpoints containing data identity,
configuration, methods, software versions, and artifacts.

Build and verify the curated public package:

```bash
python scripts/build_public_package.py
python scripts/build_public_package.py --verify
```

## Testing

The test suite covers REAL-versus-synthetic isolation, differential expression,
survival and adjusted Cox models, PH diagnostics, planning/replanning,
checkpoint persistence, verifier behavior, Gradio output contracts, and public
package secret/forbidden-file validation.

```bash
python -m pytest
python -m compileall -q src app
python scripts/build_public_package.py
python scripts/build_public_package.py --verify
```

Network-dependent and expensive live checks remain opt-in; default tests mock
external services when network behavior is not under test.

## Limitations

- LUAD, BRCA, and KIRC are the tested initial cancer scope, but not every TCGA
  cancer has a live end-to-end acceptance run.
- GEO metadata, phenotype ambiguity, annotation gaps, or supplementary formats
  can prevent automated replication; a suitable cohort may not exist.
- Adjusted Cox models use complete cases and may be ineligible because of event
  counts, missing covariates, singular designs, or convergence failures.
- Expensive uncached REAL analyses depend on external APIs and available memory,
  compute, network, and storage. Free hosted environments may be insufficient.
- Checkpoint persistence across hosted restarts requires configured persistent
  storage.
- Computational associations and evidence tiers require independent scientific
  review and validation.

## Research-Use Boundary

OncoLncAI is intended for computational research and hypothesis generation.
Results may describe a computational biomarker candidate, prognostic association,
adjusted prognostic association, literature-supported evidence, or externally
replicated computational evidence. They do not establish clinical validation,
clinical utility, diagnostic approval, or treatment recommendations.

See [architecture](docs/architecture.md),
[scientific methodology](docs/scientific_methodology.md),
[tool contracts](docs/tool_contracts.md), [evaluation](docs/evaluation.md),
[deployment](docs/deployment.md), and
[third-party/data-source attribution](docs/third_party_and_data_sources.md).

Licensed under Apache-2.0.
