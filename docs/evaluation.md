# Evaluation Plan — OncoLnc-AgentBench

## Real-mode acceptance coverage

Mocked integration tests exercise successful real-path composition and total external
failure. They assert real mode, TCGA-LUAD resolution, real-shaped survival preparation,
PMID retention, verifier success, terminal status, manifest persistence, no pending
steps, and `skipped_after_replan` for unsupported branches. A live cached acceptance run
completed with 503 samples, 182 events and four context-filtered PubMed records.

## Strengthening regression coverage

Deterministic tests cover cancer/project mismatch rejection, malformed-plan repair,
invalid-replan repair, partial completion after an independent tool failure, automatic
typed TCGA survival preparation, proportional-hazards diagnostic reporting, and UI
view-model separation of unavailable versus negative evidence. These cases complement
AgentBench v1; fixture benchmark scores remain evaluator self-tests, not live-model
quality claims. A versioned live planner/retrieval baseline is still required before
reporting empirical recovery or relevance rates.

## Purpose

Evaluate whether the system behaves correctly as an agent, not merely whether the final prose looks plausible.

## Initial benchmark scope

Start with a small curated benchmark covering:
- LUAD biomarker discovery,
- BRCA biomarker discovery,
- KIRC biomarker discovery,
- inadequate normal-sample case,
- ambiguous cancer-name case,
- invalid lncRNA case,
- conflicting-literature case,
- API failure case,
- missing survival-data case.

## Retrieval metrics

- Recall@5
- Recall@10
- MRR or nDCG
- citation precision

## Evidence extraction metrics

For fields such as:
- lncRNA,
- cancer,
- biomarker role,
- expression direction,
- outcome,
- association direction,
- evidence type.

Measure:
- precision,
- recall,
- F1.

## Agent-behavior metrics

- intent classification accuracy,
- plan validity,
- tool-selection accuracy,
- tool-argument accuracy,
- workflow-completion rate,
- replanning success rate,
- recovery rate,
- unnecessary-tool-call rate.

## Scientific output metrics

- numerical accuracy,
- DE-result agreement,
- survival-result agreement,
- evidence-attribution accuracy,
- unsupported-claim rate,
- contradiction-retention rate.

## Efficiency metrics

- LLM calls per task,
- input tokens per task,
- output tokens per task,
- local vs cloud calls,
- latency,
- cloud cost if applicable.

## Required ablations later

Compare:

1. literature-only RAG
2. transcriptomics-only workflow
3. integrated literature + TCGA agent
4. integrated + GEO validation
5. optional integrated + specialized lncRNA model

For evidence extraction compare:

1. prompt-only
2. RAG + general LLM
3. RAG + fine-tuned specialist

Do not perform fine-tuning until a baseline benchmark exists.

## Replanning test

At least one benchmark case must assert:

```text
Initial plan includes DE
Capability checker reports inadequate normal samples
Expected:
- DE skipped
- warning recorded
- survival retained if valid
- alternate validation path proposed
```

This is a core test of agentic behavior.

## Verification test

Inject:
- an unsupported claim,
- a modified numerical result,
- causal wording for observational evidence.

Expected:
- verifier flags all three.

## Implemented OncoLnc-AgentBench v1

Milestone 14 versions the initial benchmark at `benchmarks/v1/cases.json` with
schema version `1.0.0`. It contains six cases:

- standard LUAD evidence plus survival,
- standard BRCA evidence plus differential expression,
- standard KIRC evidence plus survival,
- LUAD with inadequate normals requiring DE to be skipped and survival retained,
- ambiguous `lung cancer` requiring input rather than a guessed project,
- a PubMed API failure requiring a structured failed outcome and no downstream tools.

`fixture_observations.json` is a deterministic self-test, not a claim about live
model performance. `observation_from_vertical_result` adapts an actual successful
vertical-slice trace without copying raw scientific inputs into the benchmark.
Failure and ambiguity observations use the same compact schema.

Deterministic case checks cover cancer resolution, workflow outcome, required and
prohibited tools, skipped steps, replan count, structured error handling,
verification, provenance, and bounded model/tool call budgets.

Reported agent metrics:

- overall case pass rate,
- cancer-resolution and workflow-outcome accuracy,
- tool-selection accuracy,
- replanning, failure-handling, and ambiguity-handling success rates,
- verification accuracy,
- provenance-retention rate.

Reported efficiency metrics:

- total and average model calls,
- total and average tool calls,
- call-budget compliance,
- unnecessary-tool-call rate.

The grader uses no LLM judge. Retrieval, extraction, and scientific numerical
quality metrics require curated gold data and are not inferred from these
behavioral fixture observations. Fine-tuning must not begin until real baseline
runs have been collected against the versioned cases.

## Milestone 18 baseline gate audit

Status: **provisional baseline established on 2026-09-14**.

The earlier audit found no usable evidence-extraction baseline. That prerequisite
has now been addressed in `benchmarks/extraction_v1` with:

- nine real PMID-bearing passages with manually curated structured labels,
- PMID-disjoint train (3), validation (2), and frozen test (4) partitions,
- saved prompt-only Ollama `qwen2.5-coder:14b` predictions,
- deterministic atomic field matching,
- test precision **1.0000**, recall **0.2308**, and F1 **0.3750**,
- four saved test failure cases, including one invalid structured response.

OncoLnc-AgentBench v1 measures agent behavior and call efficiency. Its fixture
observations are evaluator self-tests and cannot be repurposed as supervised
extraction labels or baseline model predictions.

The corpus addendum now provides 342 balanced deterministic-silver training
examples and 42 validation examples across LUAD, BRCA, and KIRC, while leaving the four
manually curated baseline test PMIDs frozen. This meets the explicit minimum for
a pilot schema/entity-grounding LoRA experiment. Silver labels intentionally
cover only verified lncRNA identity and cancer context; richer evidence type,
direction, role, and experimental-validation supervision still requires expert
annotation. The current low baseline recall provides a concrete target for
improvement rather than a fabricated fine-tuning claim.

### Milestone 18 frozen comparison

The requested pilot used `Qwen/Qwen2.5-Coder-1.5B-Instruct` with NF4 4-bit
quantization and LoRA (rank 8, alpha 16, dropout 0.05). Two epochs trained
9,232,384 adapter parameters; validation loss decreased from 0.07474 to 0.06658.
Only adapter, tokenizer, configuration, and metric artifacts were saved.

The untouched four-example test set produced:

| Extractor | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Frozen prompt-only Ollama `qwen2.5-coder:14b` | 1.0000 | 0.2308 | 0.3750 |
| QLoRA `Qwen2.5-Coder-1.5B-Instruct` | 0.0000 | 0.0000 | 0.0000 |

The fine-tuned model emitted a valid empty batch for every test example. Its F1
delta is -0.375, so it is not selected or wired into default application
configuration. This negative result is consistent with the corpus limitation:
silver labels primarily supervise lncRNA/cancer grounding and contain no expert
coverage adequate for the richer frozen-test evidence fields. The four-example
gold set is also too small for a general performance claim.
