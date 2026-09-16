# Tool Contracts

## Real analysis coordinator

`RealAnalysisRequest` requires a run ID, question, resolvable cancer, candidate symbol,
and Ensembl gene ID. `RealAnalysisResult` exposes real-data mode, terminal status,
capabilities, PubMed records, deterministic survival output, evidence tier, verifier,
workflow steps, limitations, provenance, manifest, and report. Network failures remain
failed steps; they never trigger synthetic fallback.

## Strengthened validated-action boundary

Every proposed action must name an allow-listed tool and validate against that tool's
exact Pydantic input schema. Planning also rejects a TCGA project inconsistent with
resolved disease and a PubMed cancer term inconsistent with the canonical cancer.
Unknown arguments and unsupported parameters fail because request schemas forbid
extras. Rejected calls are never executed.

`VerticalSliceRequest.tcga_survival_inputs` maps compact input IDs to
`TCGASurvivalDataRequest`. At execution, the existing cohort-preparation client creates
a checksum-bearing `SurvivalRequest`; only then does the deterministic survival tool
run. Direct prevalidated `survival_inputs` remain supported for tests and offline use.

## General contract

Every tool should have:
- explicit name,
- clear description,
- typed input,
- typed output,
- deterministic behavior where possible,
- provenance,
- structured warnings/errors,
- mockable external dependencies.

Suggested result wrapper:

```python
class ToolResult[T](BaseModel):
    success: bool
    data: T | None = None
    warnings: list[str] = []
    error_code: str | None = None
    error_message: str | None = None
    provenance: dict = {}
```

Do not return only free-form prose.

---

## `resolve_cancer`

Input:

```python
CancerResolveRequest(name: str)
```

Output:

```python
CancerResolution(
    canonical_name: str,
    abbreviation: str,
    tcga_project_id: str | None,
    synonyms: list[str],
    primary_site: str | None,
    confidence: float,
    ambiguous: bool,
    alternatives: list[CancerCandidate],
)
```

Behavior:
1. exact mapping,
2. normalized alias mapping,
3. fuzzy matching,
4. ambiguity result,
5. optional LLM fallback only if explicitly invoked.

---

## `search_pubmed`

Input:

```python
PubMedSearchRequest(
    cancer: str,
    concepts: list[str],
    max_results: int = 100,
)
```

Output records:
- PMID
- title
- abstract
- journal
- publication year
- authors
- query
- retrieval timestamp

Must:
- retry 429/5xx,
- deduplicate PMID,
- cache query results,
- preserve query provenance.

---

## `extract_literature_evidence`

Input:
- source text/chunks,
- source metadata,
- cancer context.

Output:

```python
EvidenceRecord(
    lncrna: str,
    cancer: str,
    pmid: str,
    evidence_type: str | None,
    biomarker_role: str | None,
    expression_direction: str | None,
    outcome: str | None,
    association_direction: str | None,
    sample_type: str | None,
    cohort_size: int | None,
    experimental_validation: bool | None,
    supporting_text: str,
    confidence: float,
)
```

LLM use is allowed because semantic extraction is required.

Schema validation is mandatory.

---

## `inspect_tcga_capabilities`

Input:

```python
TCCapabilityRequest(project_id: str)
```

Output:

```python
DataCapabilities(
    project_id: str,
    rna_seq_available: bool,
    clinical_available: bool,
    survival_available: bool,
    tumor_samples: int,
    normal_samples: int,
    de_recommended: bool,
    survival_recommended: bool,
    reasons: list[str],
)
```

This tool must be deterministic.

---

## `run_differential_expression`

Input:

```python
DifferentialExpressionRequest(
    project_id: str,
    samples: list[ExpressionSample],
    features: list[CountFeature],
    case_definition: str = "Primary Tumor",
    control_definition: str = "Solid Tissue Normal",
    candidate_genes: list[str] = [],
)
```

Samples have unique IDs and explicit `case` or `control` labels. Each feature
contains one aligned vector of non-negative integer raw counts. Every sample
must have a positive complete-library size. Default cohort thresholds are 20
cases and 10 controls.

Output per feature:
- identifier,
- log2 fold change,
- p-value,
- adjusted p-value,
- sample counts,
- method metadata.

Milestone 10 behavior:
- complete-library CPM normalization,
- `log2(CPM + pseudocount)` transformation,
- raw-count filtering before testing,
- case-versus-control log2 fold change from mean CPM,
- two-sided Welch t-test on log-CPM,
- Benjamini-Hochberg correction across the declared tested scope,
- explicit filtered-feature reasons,
- count and annotation provenance.

No LLM computation.

## `prepare_tcga_survival_cohort`

Input:

```python
TCGASurvivalDataRequest(
    project_id="TCGA-LUAD",
    gene="NEAT1",
    ensembl_gene_id="ENSG00000245532",
    endpoint="OS",
)
```

Output includes aligned `SurvivalSample` records plus source URLs, retrieval
time, SHA-256 hashes, expression scale, source/filtered counts, duplicate
aliquot count, missing endpoint count, event count, and provenance.

The adapter downloads through an injectable transport, caches atomically,
filters primary tumors, selects one aliquot per patient, and aligns expression
with curated survival endpoints. It prepares inputs only; all statistics remain
inside `run_survival_analysis`.

---

## `run_survival_analysis`

Input:

```python
SurvivalRequest(
    project_id: str,
    gene: str,
    samples: list[SurvivalSample],
    endpoint: Literal["OS", "DSS", "PFI"] = "OS",
    grouping_method: str = "median",
)
```

Each `SurvivalSample` contains a unique sample ID, positive endpoint time,
binary event indicator, and finite expression value. The request also preserves
minimum sample/event thresholds, fixed alpha 0.05, cohort source, and expression source.

Output:
- HR,
- 95% CI,
- Cox p-value,
- log-rank p-value,
- event count,
- sample count,
- grouping metadata,
- model diagnostics where practical.

Milestone 9 behavior:
- median split with values tied at the median assigned low,
- Kaplan-Meier estimates for low/high groups,
- two-sided log-rank test,
- univariate Cox PH for the high-expression indicator,
- Breslow handling of tied event times,
- structured failure when the model is singular or cannot converge,
- provenance for clinical and expression inputs.

No LLM computation.

---

## `search_geo`

Input: `GEOSearchRequest` with cancer, tissue, aliases, allowed assays, minimum
case/control counts, annotation requirement, and a bounded result limit.

Output: `ToolResult[GEOSearchResult]` containing every normalized candidate,
six per-dataset criterion decisions, deterministically ranked suitable datasets,
an explicit `no_suitable_validation_dataset` flag, and accession/query
provenance.

Selection requires all of:
- cancer/disease match,
- tissue match,
- allowed assay and recorded platform,
- case and control availability,
- configured minimum group sizes,
- compatible lncRNA annotation.

Unknown metadata fails closed. Optional semantic review may resolve only an
ambiguous disease or tissue match. It cannot override assay, group, sample-size,
or annotation failures, and title similarity alone is never sufficient.

---

## `verify_claims`

Prefer multiple checks:
1. deterministic numeric comparison,
2. identifier validation,
3. source/provenance existence,
4. semantic claim-support model only where needed.

Output:

```python
VerificationReport(
    passed: bool,
    unsupported_claims: list[...],
    numeric_mismatches: list[...],
    missing_sources: list[...],
    causal_language_flags: list[...],
    warnings: list[str],
)
```

---

## `prioritize_evidence`

Input: `EvidenceFusionRequest` containing a candidate, cancer, optional
literature/TCGA/survival/GEO observations, and `EvidenceFusionConfig`.

Output: `ToolResult[EvidencePrioritizationResult]` containing:
- the deterministic evidence prioritization score on a 0–100 scale,
- every component's availability, 0–1 score, configured weight, weighted points,
  and rationale,
- evidence coverage and explicitly missing components,
- consistency label, agreement/contradiction counts, and comparison details,
- immutable scoring configuration and combined provenance.

Missing components are not negative evidence and are excluded from the score
denominator. An available but nonsignificant statistical result scores zero.
Any optional LLM synthesis receives the immutable result and returns narrative
fields only; it cannot generate or modify a numerical score.

---

## Tool registry

Registry should expose only allowed tools.

The planner must not execute arbitrary shell or Python generated by the model as a scientific-analysis substitute.

## `predict_lncrna_localization`

Input: `LocalizationRequest(lncrna_id, sequence=None)`.

Output: `ToolResult[LocalizationResult]` containing a validated predicted
compartment, optional normalized compartment probabilities, optional sequence
length, model name/version/source/artifact URI, explicit interpretation limits,
and model provenance in the common result envelope.

The backend is optional and injectable. `CallableLocalizationBackend` adapts an
existing sequence prediction function; `PrecomputedLocalizationBackend` reads
validated saved outputs. `LocalProjectLocalizationBackend` verifies the existing
project's version and model SHA-256 before lazily loading its public
`predict_probabilities` function. With no configured backend, the tool is non-executable
in the default registry and direct invocation returns
`localization_unavailable`. A missing precomputed identifier returns
`localization_not_found` rather than guessing.

Localization is not an input to `prioritize_evidence`, no compartment receives
a score or reward, and probabilities are omitted when their source does not
provide them. No LLM is used.

---

## `run_vertical_slice`

Input:

```python
VerticalSliceRequest(
    run_id: str,
    research_question: str,
    cancer_name: str,
    survival_inputs: dict[str, SurvivalRequest],
    differential_expression_inputs: dict[str, DifferentialExpressionRequest],
)
```

The planner sees only compact `AnalysisInputReference(input_id=...)` arguments.
The workflow registry resolves those IDs to the caller-supplied typed analysis
inputs and invokes the existing deterministic tools; raw cohorts and matrices
are not placed in the planner prompt.

Output includes the completed `ResearchState`, initial and revised plans,
in-memory agent events, retrieved papers, extracted evidence, statistical
results, verification report, and a `GroundedFinalResult`. Completion requires:

- unambiguous cancer resolution,
- one PubMed retrieval action,
- source-grounded extracted evidence,
- TCGA capability inspection,
- at least one capability-supported statistical analysis,
- successful claim verification.

Every final claim carries source IDs and any reported numerical values. The
final result must retain limitations and the research-only disclaimer.
## Genome-wide TCGA and downstream contracts

- `GenomeWideDataRequest → PreparedGenomeWideCohort` preserves sample metadata, normalized
  Ensembl IDs, symbols, biotypes, source checksums, preprocessing, and duplicate counts.
- `run_real_genome_wide_de(PreparedGenomeWideCohort)` is the only adapter into the existing
  `run_differential_expression` calculation.
- `CandidateDiscoveryResult` exposes every selection threshold and the eligible/retained counts.
- `AdjustedCoxRequest → AdjustedCoxResult` reports complete-case N, events, expression HR and CI,
  p/FDR, covariates, convergence, and term-level PH diagnostics.
- `GEOReplicationRequest → GEOReplicationResult` accepts only an already compatible annotated
  matrix and reports mapped/unmapped candidates, effect, p/FDR, and TCGA direction consistency.
- Major coordinator branches expose implementation, data availability, input validity,
  statistical eligibility, explicit status, and reason separately.

`DifferentialExpressionRequest.method` supports `AUTO`, `PYDESEQ2`, and `WELCH` and requires an
explicit `input_data_type`. Every common result configuration records the selected/requested
method, input representation and transform, normalization/filtering, case/control N, and tested
gene/lncRNA counts. `PYDESEQ2` rejects transformed continuous input at validation time.

`GEOExpressionClient` supports standardized Series Matrix and GPL annotation files. It preserves
titles, source names, raw characteristics, sample accessions, platform, source URLs, ambiguous
mappings, and missing phenotype fields. Only an `ELIGIBLE` cohort produces a
`GEOReplicationRequest`.

`GDCRawCountRequest → PreparedGDCRawCountCohort` preserves every selected GDC file UUID, sample
and case ID, sample type, upstream checksum/size, exact query, assembled-matrix checksum, cache
path, and purpose-specific source metadata. Its `de_request()` always declares
`RAW_INTEGER_COUNTS`, transform `none`, native GDC distribution, and `AUTO` routing.

The curated GEO registry lives at `src/oncolncai/data/geo_cohorts.json`. Only entries marked
`verified` can resolve unknown automatic phenotype labels, and only the configured field/value
mapping is executable.
