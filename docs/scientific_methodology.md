# Scientific Methodology

## Deployed real-data candidate validation

The deployed supported path validates one supplied lncRNA symbol plus Ensembl gene ID.
It deterministically filters primary-tumor expression, aligns one sample per patient to
PanCanAtlas survival endpoints, runs median-split KM/log-rank/univariate Cox plus PH
diagnostics, and retrieves bounded PubMed metadata. PubMed display applies a conservative
cancer/candidate context filter to reduce gene-name collisions. This is candidate
validation, not genome-wide discovery, clinical validation, or independent replication.

## Strengthening addendum — validated real-input and survival boundary

The supported TCGA survival path can accept a typed preparation request and invoke the
existing deterministic Xena/GDC cohort builder: primary-tumor filtering, one aliquot per
patient, endpoint validation, expression/clinical alignment, checksums, and then the
statistical tool. Sample-level tables never enter an LLM prompt.

Cox output includes a proportional-hazards diagnostic based on the Spearman time trend
of Schoenfeld residuals for the fitted high-versus-low term. A significant result
produces an explicit warning; a non-estimable diagnostic is reported rather than
imputed. The analysis remains univariate and median-split and cannot support an
"independent prognostic biomarker" claim. Covariate adjustment remains unavailable
until consistently defined clinical covariates are supplied.

The CPM/log2 plus Welch DE implementation remains a deterministic candidate-screening
method, not a count-aware publication-grade RNA-seq model. No DE result is forced when
normal samples are inadequate.

## Scope

OncoLncAI prioritizes candidate lncRNAs by combining:
1. literature evidence,
2. public RNA-seq evidence,
3. prognostic evidence,
4. optional external replication,
5. optional biological context.

This is evidence prioritization, not clinical validation.

## Literature evidence

Extract structured claims with provenance:

- lncRNA
- cancer
- PMID/source
- evidence type
- biomarker role
- expression direction
- outcome
- association direction
- sample type
- cohort size if reported
- experimental validation
- supporting passage
- confidence

Never detach an extracted claim from its source.

## Candidate discovery

A candidate may enter the list because:
- it appears in multiple relevant studies,
- it has strong clinical/mechanistic evidence,
- it is explicitly described as diagnostic/prognostic,
- it has consistent independent support.

The final ranking should not be an opaque LLM preference.

## TCGA capability checks

Before statistical analysis, inspect:
- RNA-seq availability,
- clinical data availability,
- survival endpoint availability,
- tumor sample count,
- normal sample count,
- annotation compatibility.

If normal samples are inadequate:
- do not force TCGA tumor-vs-normal DE,
- record the limitation,
- permit survival analysis when valid,
- optionally consider GEO later.

Do not automatically combine TCGA and GTEx in v1.

## Differential expression

Use deterministic statistical code.

Minimum reporting:
- case/control definition,
- sample counts,
- effect size,
- p-value,
- adjusted p-value,
- method,
- filtering/normalization assumptions.

The LLM may explain results but may not generate them.

Milestone 10 accepts an aligned raw, non-negative integer count matrix and
explicit case/control sample labels. Production defaults require at least 20
case and 10 control samples, consistent with the capability policy.

The initial deterministic method is:

1. calculate each complete sample library size across all supplied features,
2. normalize counts to counts per million (CPM),
3. transform to `log2(CPM + 1)` for the two-sided Welch test,
4. retain a feature when the raw-count threshold is met in the configured
   number of samples in either group,
5. calculate `log2((mean case CPM + 1) / (mean control CPM + 1))`,
6. apply Benjamini-Hochberg correction across every tested feature in the
   declared analysis scope.

Filtering occurs before hypothesis testing and multiple-testing correction.
Candidate-restricted analysis is allowed, but its correction scope must be
reported as the candidate subset rather than transcriptome-wide. Results
preserve case/control definitions, sample and feature counts, excluded features
and reasons, normalization/test/filter configuration, effect sizes, raw and
adjusted p-values, and count/annotation provenance.

This lightweight Welch-on-log-CPM method is suitable for the initial software
vertical slice and deterministic candidate checks. It is not a replacement for
count-aware models such as DESeq2 or edgeR in a publication-grade analysis,
especially for complex designs, batch adjustment, or small cohorts.

## Survival analysis

Support:
- Kaplan-Meier curves,
- log-rank test,
- Cox proportional hazards,
- hazard ratio,
- 95% confidence interval,
- p-value,
- optional FDR for multi-candidate testing.

Use language such as:
- “associated with worse overall survival”
- “prognostically relevant in this cohort”

Avoid causal language.

Milestone 9 uses aligned per-sample expression, endpoint time, and binary event
status. Inputs must have unique sample identifiers, finite expression, positive
follow-up time, sufficient samples/events, and two non-empty expression groups.

Initial grouping is a deterministic median split:

- high: expression strictly greater than the cohort median,
- low: expression less than or equal to the median,
- values tied at the median remain in the low group.

Kaplan-Meier curves report risk sets, events, censoring, and survival
probabilities at observed times. The log-rank test compares the two curves. The
univariate Cox model uses the high-expression indicator as its sole covariate,
so the hazard ratio is **high versus low**, with Breslow handling of tied event
times. Report the configured endpoint, alpha, sample/event counts, HR, CI,
Cox p-value, log-rank p-value, convergence diagnostics, and data provenance.
Milestone 9 fixes alpha at 0.05 so the reported interval is always 95%.

The initial Cox model is unadjusted. Its result is a cohort association and must
not be interpreted as causation, independent prognostic value, or clinical
validation. Proportional-hazards diagnostics and clinical-covariate adjustment
are not yet implemented.

### Real-data validation addendum

The pinned Milestone 9 validation uses the public GDC-derived TCGA-LUAD
STAR-count matrix distributed by the UCSC Xena GDC hub and the curated TCGA
PanCanAtlas survival endpoint table. NEAT1 is resolved by versionless Ensembl ID
`ENSG00000245532`; the version suffix in the matrix row is ignored only for the
exact stable-ID comparison.

Preparation is deterministic:

1. retain TCGA sample type `01` primary tumors,
2. choose the lexicographically first aliquot when a patient has duplicates,
3. join expression to survival by the first three TCGA barcode fields,
4. use the curated `OS` and `OS.time` values,
5. exclude missing, non-binary, or non-positive endpoints,
6. preserve source URLs, retrieval time, SHA-256 hashes, exclusion counts, gene
   identifier, and source-provided expression-scale label.

The aligned real-data fixture is a reproducibility snapshot, while synthetic
fixtures remain the authoritative unit tests for known statistical behavior.
The expression matrix distributor's transformation is retained by label rather
than reinterpreted as raw counts. Median grouping is invariant to monotonic
log transformations, but other modeling choices may not be.

## Independent validation

GEO may be added after the TCGA vertical slice works.

Dataset suitability must consider:
- correct disease,
- correct tissue,
- relevant assay,
- sample groups,
- sample size,
- annotation,
- platform compatibility.

Do not select validation datasets solely by title similarity.

Milestone 15 applies these as six explicit, inspectable checks. A dataset is
suitable only when every check passes. Unknown disease/tissue metadata may be
sent for bounded semantic review, but unknown group counts, assay, or annotation
remain failures. Ranking among eligible datasets uses the number of passed
criteria, then total case/control sample count, then accession for stable ties.
The valid outcome may be that no suitable independent validation dataset exists;
in that case validation is not forced.

## Evidence consistency

Conflicting evidence must remain visible.

Example:

```text
Study A: adverse prognosis
Study B: no significant association
TCGA: weak association
```

Output should explicitly characterize evidence as:
- consistent,
- mostly consistent,
- mixed,
- conflicting,
- insufficient.

## Evidence prioritization score

If a score is used, expose components.

Possible dimensions:

```text
L = literature support
D = differential-expression evidence
P = prognostic evidence
R = independent replication
M = mechanistic evidence
Q = study quality
C = cross-study consistency
```

Example:

```text
Score = wL*L + wD*D + wP*P + wR*R + wM*M + wQ*Q + wC*C
```

Milestone 16 calls this an **evidence prioritization score**, never a
probability that a candidate is a true biomarker. Its implemented components are
literature support, TCGA differential expression, survival association, GEO
replication, and cross-source consistency. Each component is bounded to `[0, 1]`
and its configured weight and weighted points are exposed.

The final score is `100 * sum(available weighted points) / sum(available
weights)`. Missing evidence is excluded from both sums and reported separately
through `missing_evidence` and `evidence_coverage`; it is not converted to a zero.
Observed nonsignificant evidence is available evidence and therefore receives a
zero for that component. Expression and prognosis directions are compared only
within their respective dimensions. Agreement counts, contradiction counts,
and pair-level details remain visible.

Literature support uses a configurable saturation count of unique PMIDs. TCGA and GEO
expression components require adjusted significance and scale with absolute
log2 fold-change. Survival requires both Cox and log-rank significance and
scales with absolute log hazard ratio. Thresholds, effect scales, and weights
are configuration, not learned parameters. An optional narrative model can
explain the completed result through a schema that contains no score field.

Call this an **evidence prioritization score**.

Do not describe it as a probability that the candidate is a true biomarker.

## lncRNA localization context

Subcellular localization may help interpret plausible biological roles or guide
follow-up experiments, but it does not establish differential expression,
prognostic association, diagnostic performance, mechanism, or clinical utility.
Milestone 19 therefore reports localization separately and does not feed it into
the evidence prioritization score. No compartment is intrinsically rewarded.

Probabilities are exposed only when the wrapped predictor or precomputed record
actually supplies them. They are validated to lie in `[0, 1]` and sum to one,
and remain model-output probabilities rather than biological certainty. Every
result identifies the model, version, source, and optional artifact location.
The integrated Random Forest v2.0.0 was trained on 4,526 unique,
non-conflicting sequences from the dissertation RCI canonical datasets. Its
recorded held-out test metrics are accuracy 0.6777, balanced accuracy 0.6211,
and ROC AUC 0.7053. This is one internal transcript-level split, not independent
external or gene-grouped validation, so its output remains biological context
rather than biomarker evidence or clinical validation.

## Initial cancer validation set

Benchmark initial implementations on:
- TCGA-LUAD
- TCGA-BRCA
- TCGA-KIRC

Design reusable functions rather than cancer-specific code.
## Genome-wide real-data methodology

- Source expression is the UCSC Xena copy of GDC-harmonized TCGA STAR counts. Because that
  matrix is encoded as `log2(count+1)`, preparation deterministically applies `round(2**x-1)`
  before the count-based DE engine performs CPM/log2 normalization.
- TCGA sample type `01` is primary tumor and `11` is solid-tissue normal. Replicate aliquots
  within the same patient/group are resolved by selecting the lexicographically first barcode;
  paired tumor and normal biospecimens remain separate observations.
- GENCODE v36 `gene_type` controls lncRNA inclusion; Ensembl version suffixes are removed for
  joins, and duplicate normalized gene rows are removed deterministically and counted.
- Candidate discovery requires configured FDR, absolute log2 fold-change, and mean-expression
  thresholds before top-N truncation. No LLM selects candidates.
- Survival screens receive BH correction across candidates. Adjusted Cox uses standardized
  expression and age plus encoded sex and pathological stage when complete cases and events are
  adequate. Schoenfeld-residual time-trend diagnostics propagate violations to tiering/reporting.
- GEO is not forced: metadata suitability must pass disease, tissue, assay/platform,
  case/control, sample-size, and annotation checks. If no series passes, external replication is
  `NOT_AVAILABLE`; if a series passes but its matrix cannot be routed, it is
  `INTEGRATION_INCOMPLETE`.

### Prompt 1.5 expression audit and DE routing

The authoritative sidecar for `TCGA-LUAD.star_counts.tsv` declares label `STAR - Counts`,
unit `log2(count+1)`, Xena GDC ETL wrangling that averages multiple vial/portion/analyte/aliquot
inputs from the same sample and then transforms them, dataset version `05-10-2024`, and GDC
Data Release 40.0 as the upstream release. Therefore the distributed artifact is
`XENA_LOG2_STAR_COUNTS`, not untouched raw integer counts.

`differential_expression.py` remains the public interface. `AUTO` selects PyDESeq2 only for
inputs explicitly declared `RAW_INTEGER_COUNTS`; it selects the existing CPM/log2-Welch method
for Xena-transformed or other normalized continuous data. An unavailable optional count backend
fails explicitly and never silently falls back. A DESeq2-versus-Welch LUAD comparison is not
scientifically valid for the one distributed representation and is therefore not performed.

Complete-case adjusted Cox remains the primary adjusted analysis. Reporting now includes initial
survival-eligible N, complete-case N, excluded N, per-covariate missingness, events, parameter
count, events per parameter, convergence, HR/CI/p/FDR, and PH diagnostics. Models fail eligibility
when events per parameter are inadequate. No automatic imputation, penalization, stratification,
or time-varying model was introduced.

### Native GDC raw-count production DE

Prompt 1.75 introduced a separate inference-grade source for DE. `GDCRawCountClient` retrieves
open GDC `Gene Expression Quantification / STAR - Counts` files, validates integer semantics, and
feeds the untransformed `unstranded` column to PyDESeq2. The live LUAD acceptance cohort uses a
deterministic bounded subset of 50 primary tumors and 50 solid-tissue normals. This bound limits
portfolio-runtime download cost and is disclosed in the manifest; it is not presented as the full
TCGA cohort. Xena continues to provide transformed expression for survival.

Exploratory transformed-expression Welch analysis remains available only through explicit
`allow_exploratory_continuous_de_fallback`; production does not silently downgrade when native
counts or PyDESeq2 are unavailable.
