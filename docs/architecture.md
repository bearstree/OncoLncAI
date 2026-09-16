# Architecture

## Real-data application path

`RealAnalysisRunner` composes existing cancer resolution, GDC capability, TCGA/Xena
survival preparation, deterministic statistics, PubMed, verification, provenance, and
manifest contracts. It is separate from `run_configured_agent_demo`; no real failure can
fall through to a synthetic result. Workflow steps end as completed, failed, or
`skipped_after_replan`, never unreachable pending work. The Gradio progress callback
publishes the current numbered tool step while the final workflow table preserves the
observation and decision.

## Strengthened validation, recovery, and UI boundary

Planner output crosses two deterministic gates before execution: the registered
Pydantic request schema and resolved-context consistency checks. TCGA project IDs must
match the resolved disease and PubMed cancer arguments must match its canonical name.
One rejected proposal may receive one bounded repair call; an optional fallback provider
may be configured afterward. Scientifically meaningful identifiers are never silently
rewritten. Repair and rejection are returned as warnings and recorded by the vertical
slice trace.

The agent loop records failed actions independently and continues safe remaining
actions. Completed observations and provenance survive, producing a partial result.
Capability-driven replanning uses the same bounded repair rule and retains or explicitly
skips every remaining step.

The Gradio layer is a read-only projection of result models. It shows question and
resolution, plan status, concise trace events, capability observations, replanning,
evidence/computation boundaries, verification, limitations, and final outcome. It
performs no scientific computation and displays no hidden reasoning.

## Objective

OncoLncAI is an evidence-to-validation agent for cancer lncRNA biomarker discovery.

The system should answer questions such as:

> Which lncRNAs are promising biomarkers for breast cancer, what evidence supports them, and are they differentially expressed or prognostically relevant in public RNA-seq datasets?

## Architectural principle

Use LLMs for semantic reasoning; use deterministic tools for data access and computation.

```text
User
  ↓
Intent interpretation
  ↓
Supervisor / Planner
  ↓
ResearchState
  ↓
Tool Registry
  ├── Cancer Resolver
  ├── PubMed/PMC
  ├── Evidence Extractor
  ├── GDC/TCGA Capability Checker
  ├── Differential Expression
  ├── Survival
  ├── GEO Validation
  ├── Enrichment
  └── Optional lncRNA Localization
  ↓
Observation
  ↓
State Update
  ↓
Replan if needed
  ↓
Verifier
  ↓
Evidence Fusion
  ↓
Final report
```

## Agent loop

The core control loop is:

```text
Goal
→ Plan
→ Act
→ Observe
→ Update state
→ Replan if necessary
→ Verify
→ Finish
```

The application must not use the same fixed action sequence for every request.

## Conceptual roles

### Supervisor / Planner
Responsibilities:
- classify intent,
- create a minimal plan,
- select tools,
- decide whether observations require replanning,
- stop when the goal is sufficiently answered.

### Literature Specialist
Responsibilities:
- produce retrieval queries,
- interpret retrieved biomedical text,
- extract structured lncRNA evidence.

May be implemented as a prompt mode rather than a separate long-lived agent.

### Genomics Specialist
Responsibilities:
- choose which deterministic analyses are appropriate based on available cohort capabilities,
- interpret validated statistical outputs.

### Verifier
Responsibilities:
- check claim/source alignment,
- compare reported numbers with tool output,
- surface limitations and contradictions.

## Tool boundary

Tools must be callable independently of the agent.

Example:

```python
result = run_survival_analysis(project_id="TCGA-LUAD", gene="MALAT1")
```

The LLM decides whether/why to call the tool. The tool performs the computation.

## Model-provider boundary

All LLM use should go through a small provider abstraction.

Initial providers:
- OpenAI
- Ollama
- Mock

Do not scatter direct model SDK calls across agent modules.

Milestone 17 implements this boundary with `MockLLMProvider`,
`OpenAICompatibleProvider`, and `OllamaProvider`. All expose the same
`generate_structured(prompt, output_schema)` method. Configuration and
`create_llm_provider` choose an adapter; planners, extractors, replanners,
semantic verifiers, and synthesizers do not branch on vendor.

Both HTTP providers send the requested Pydantic JSON schema, use a configured
timeout, and translate network or response failures into `LLMProviderError`.
The HTTP transport is injectable so tests never require a vendor service.

## Repository target

```text
oncolncai/
├── app/
│   ├── api/
│   └── ui/
├── agents/
│   ├── supervisor.py
│   ├── verifier.py
│   └── state.py
├── tools/
├── rag/
├── models/
├── evaluation/
├── workflows/
├── config/
├── tests/
├── examples/
├── checkpoints/
└── docs/
```

Keep the structure smaller if modules are empty.

## v1 vertical slice

Portfolio v1 should first prove:

```text
Question
→ resolve cancer
→ search PubMed
→ extract lncRNA evidence
→ inspect TCGA capabilities
→ choose one valid analysis
→ verify claims
→ produce report
```

The key demonstration is adaptive replanning.

Example:

```text
Observation: insufficient normal TCGA samples
→ skip tumor-vs-normal DE
→ retain survival analysis
→ flag limitation
→ optionally recommend later GEO validation
```

### Implemented Milestone 12 coordinator

`VerticalSliceRunner` is a thin coordinator, not a replacement for the tool
modules. It resolves the cancer, obtains and validates a bounded plan, retrieves
PubMed records, extracts evidence from the returned abstracts, and delegates
capability-aware analysis execution to `AgentLoop`. The existing verifier checks
all final claims before a deterministic grounded result is rendered.

Plans carry compact `AnalysisInputReference` values. Prepared survival cohorts
and count matrices remain typed Python inputs outside the model context; registry
handlers resolve a reference and call the existing deterministic statistical
tool. This prevents raw sample tables from being sent to the planner.

The coordinator preserves the initial plan and the capability-revised executable
plan. A completed workflow is marked complete only after at least one valid
statistical analysis and successful verification.

## Execution tracing

Milestone 13 adds an `ExecutionTrace` alongside—but not inside—`ResearchState`.
It records the ordered operational history: run boundaries, bounded model calls,
tool calls, selected actions, observations, replanning, and verification. Each
event has a timestamp, component, operation, status, compact summary, optional
step/tool identity, and relevant provenance.

Traces deliberately exclude prompts, model responses, credentials, raw source
payloads, expression matrices, sample-level survival data, and hidden reasoning.
They can be returned in memory or atomically persisted as schema-validated JSON.
Benchmark scoring over traces belongs to Milestone 14.

## GEO validation discovery

Milestone 15 adds `GEOClient` behind a typed, injectable `GEOBackend`. The
production backend performs bounded NCBI GEO DataSets ESearch/ESummary requests;
tests and offline demonstrations inject normalized records. Dataset assessment is
deterministic and keeps the accession, metadata URL, query, retrieval timestamp,
platform, group counts, and annotation status visible.

The optional model boundary is deliberately narrow: it can resolve an otherwise
ambiguous disease/tissue meaning from a compact candidate description. It cannot
make an ineligible assay, inadequate cohort, or incompatible annotation pass.
GEO expression download and cross-study evidence fusion are not part of this
milestone.

## Deterministic evidence fusion

Milestone 16 adds `prioritize_evidence` as a typed registered tool. It consumes
compact, provenance-bearing literature, TCGA differential-expression, survival,
and GEO validation observations. Component scoring, missingness handling,
direction comparisons, contradiction counts, weighting, and aggregation are
ordinary Python. The result exposes both score and coverage so a high score
based on sparse evidence is distinguishable from a broadly supported one.

`synthesize_evidence_prioritization` is an optional narrative boundary. Its
output schema has no numerical fields, preventing a model from changing the
deterministic evidence prioritization score. Integrating this result into the
end-to-end final report remains separate from this milestone.

## Optional lncRNA localization

Milestone 19 adds `predict_lncrna_localization` as a typed optional tool. A
`CallableLocalizationBackend` wraps an existing sequence predictor without
copying or retraining it; `PrecomputedLocalizationBackend` reads validated
versioned predictions when a runtime is unavailable. Both return the same
`LocalizationPrediction` contract and preserve model name, version, source, and
artifact URI.

`LocalProjectLocalizationBackend` points directly at the existing
`Deploy_lncrna` directory, verifies that `VERSION` matches model metadata and
that the Random Forest SHA-256 matches the recorded digest, then lazily invokes
the project's public `predict_probabilities` function. The artifact is neither
copied nor retrained.

The default registry describes the capability but gives it no executable
handler until a backend is explicitly supplied. Core planning and analysis
therefore continue to work without localization. Localization remains outside
the evidence-prioritization inputs and arithmetic: nuclear, cytoplasmic, mixed,
and other compartments are biological context, not positive or negative
biomarker evidence.

## Local container boundary

Milestone 21 packages the existing agent behind a thin Gradio entry point. The
entry point calls project modules and contains no duplicate research logic. It
binds to `0.0.0.0:7860`; provider, model, endpoint, timeout, cache, and optional
credentials remain environment configuration.

The Docker build context is a strict allow-list: `src/`, `app/`, the dependency
lock, package metadata, README, Dockerfile, and `.dockerignore`. Development
instructions, documentation logs, notebooks, tests, benchmark data, training
artifacts, local model/checkpoint files, VCS metadata, caches, `.env`, and secret
files do not cross the container boundary. The image runs as an unprivileged
user and defaults to a mocked-dependency demonstration so startup and CI require
neither network access nor a live LLM.

`compose.yaml` supplies the local host-to-container mapping that image metadata
cannot enforce. It maps `127.0.0.1:7861`-facing host access to container port
7860 by default while the service continues to bind correctly to `0.0.0.0`
inside the container.

## Anti-overengineering rules

Do not add initially:
- many autonomous agents,
- distributed queues,
- Kubernetes,
- large knowledge graphs,
- raw FASTQ processing,
- complex persistent memory,
- huge vector stores,
- MCP before core tools work,
- fine-tuning before an extraction baseline exists.
## Real genome-wide discovery path

`GenomeWideAnalysisRunner` coordinates existing tools without moving numerical work into an
LLM. `TCGAGenomeWideClient` retrieves the GDC-harmonized Xena STAR-count matrix and official
GENCODE v36 annotation. The preparation layer normalizes Ensembl versions, identifies sample
types 01/11, retains the lexicographically first aliquot per patient and group, inverts the
source `log2(count+1)` representation, and retains annotated lncRNAs. It then calls
`tcga_discovery.run_real_genome_wide_de` → `differential_expression.run_differential_expression`.

Candidate selection, survival/adjusted-Cox statistics, BH correction, PH diagnostics, and tiering
are deterministic. The coordinator records observations and replans after eligibility,
candidate-bounding, GEO suitability, or model failure observations. The previous
`RealAnalysisRunner` remains the single-candidate path.

The data-source contract distinguishes `upstream_source=NCI GDC/TCGA` from
`distribution_source=UCSC Xena GDC Hub`. The Xena dataset URL, metadata-sidecar URL, checksum,
retrieval time, representation, transform, release reference, and GENCODE version/checksum are
manifest fields. A future native GDC backend can implement the same preparation contract; it is
not required by the current architecture.

GEO integration is layered: NCBI series discovery/suitability → standardized Series Matrix
retrieval → deterministic sample-phenotype parsing → GPL annotation mapping → compatibility and
candidate-coverage validation → existing replication statistics. Unsupported supplementary
formats or ambiguous phenotype/mapping states return `INTEGRATION_INCOMPLETE` or `NOT_ELIGIBLE`.

### Purpose-specific expression sources

Production no longer asks one artifact to serve incompatible statistical purposes:

- survival/inspection uses the transformed UCSC Xena GDC-hub cohort matrix;
- differential expression uses a bounded native-GDC open-access STAR-count cohort assembled by
  `GDCRawCountClient` and analyzed through the canonical DE interface with PyDESeq2.

The raw backend queries only the resolved project, selects primary tumor and solid-tissue normal
files deterministically, validates GDC size and MD5, excludes GENCODE `_PAR_Y` duplicate rows,
normalizes Ensembl versions, caches source files, assembles a gzipped lncRNA matrix, and records
SHA-256 checksums. A repeated run reuses the validated cache.
