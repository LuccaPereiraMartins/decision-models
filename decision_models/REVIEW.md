# Critical review for academic ML readers

Reviewed 23 October 2026 against `b5fa6c7`. The intended contribution is an
empirical study of financial-document routing, with explicit error/workload/cost
tradeoffs. The evidence supports that framing. It does not establish a generally
superior model family or a validated investment decision system.

## Findings and their limits

1. **The primary comparison is inconclusive on F1 superiority.** At 4k training
   documents, Laya LoRA scores 0.7843 and TF-IDF/logistic 0.7645. The paired
   Laya-minus-TF-IDF interval is [-0.0022, 0.0423]. It establishes neither
   superiority nor equivalence. At default decisions Laya skips 648 documents
   and misses 268/1,543 review references; TF-IDF skips 492 and misses 188.
   Laya's measured warm mean latency is 64.9 times higher. Preserve all three
   dimensions when describing the result. See paper Section 5.1 and Table 2.

2. **Reference validity remains the central scientific gap.** Training and
   evaluation use the same teacher, with retained three-class decisions mixed
   with fresh binary labels. The 93% consistency result comes from a balanced
   200-document training audit; it does not measure analyst agreement or the
   prevalence-weighted error rate. The paper already acknowledges this. The
   useful next study is a blinded, independently adjudicated analyst sample,
   with disagreement, class/source support and uncertainty reported.

3. **The evaluation is retrospective and narrowly sampled.** Of 2,000 test
   documents, 1,800 were inspected in the earlier study and 1,870 are SEC
   documents. A later timestamp and a new freeze do not undo prior inspection.
   The 1,967 event/near-duplicate groups are a defensible bootstrap unit for
   the recorded grouping, but do not prove independence across issuers or
   recurring templates. Confirm generalization on fresh documents with a
   prospectively fixed protocol and representative source mix. Do not select
   a new preferred recipe from this test and call that confirmation.

4. **Workload is a document-count proxy.** The paper's phrase "reading removed"
   should be interpreted as the fraction of documents skipped. It does not
   measure reading minutes, document difficulty or economic loss. Under the
   1% calibration miss-target policy, TF-IDF misses 20/1,543 test references
   (about 1.30%) and Laya misses 7/1,543 (about 0.45%), with different skip
   fractions. Neither is evidence of a population guarantee. Calibration has
   only 365 positive references, so the 1% target permits at most three observed
   misses. A future operational study needs a predeclared miss/workload objective
   and uncertainty that includes calibration fitting.

5. **Adaptation and scaling results are conditional.** Matched pre-update
   controls support improvements within the fitted recipes. Cross-family
   differences also include context, pooling, parameter counts, objectives and
   optimization budgets. Nested sample sizes change source composition and
   optimizer steps. One seed and fixed-run bootstrap intervals leave fitting
   variability unresolved. The FinBERT 25k extension is appropriately separated
   as development evidence; its selection gain is not a new test result.

6. **Full-study reproduction currently stops at arithmetic.** The two body-free
   packages reproduce saved predictions' metrics, policy decisions and recorded
   provenance. They cannot establish that the saved logits came from the stated
   model or that live recollection yields the original text. Exact fitting and
   inference need the retained snapshots and selected neural weights. The public
   TF-IDF serving artifact does support inference on supplied documents. Moreover,
   `prepare.py` extends an earlier cohort and requires its historical inputs;
   the public collectors alone do not rebuild the frozen benchmark. Before an
   artifact-evaluated submission, provide an access route for those inputs and
   weights, their reuse conditions, and a standalone fresh-data preparation
   example. No such access or license is presumed by this branch.

## Code changes in this branch

- **Repair the documented environment.** Python 3.11 installed NumPy 2.4.6 and
  SciPy 1.17.1, then serving rejected them against the frozen manifest. The
  project now requires Python 3.12+, pins the manifest's NumPy/SciPy versions,
  and documents a locked Python 3.13 environment instead of modifying system
  Python. The regenerated lock removes the unsupported Python 3.11 alternatives.
- **Keep classical computation independent.** Move the small source-family and
  deterministic sampling helpers into `schemas.py`; metrics no longer import
  preparation and MinHash. Import PyTorch only for encoder/neural paths, so
  TF-IDF inference can run in the small CPU environment.
- **Make policy selection easier to inspect.** Score each eligible threshold
  once, then apply each miss/skip constraint to the same candidate list. Keep
  the original thresholds, tie handling, review-only fallback and output fields.
  Reuse computed class F1 values instead of recomputing the same statistic.
- **Check the classifier hash before deserialization.** `LinearModel` now
  rejects a changed artifact before calling `joblib.load`. A checksum still
  requires a trusted manifest; it does not make arbitrary pickle files safe.
- **Provide a short reading path.** The study guide explains the four decision
  steps, the denominator of each metric and which files contain the essential
  method. Historical acquisition and scheduling machinery remain available.

The frozen evidence, independent verifiers, neural recipes, paper source/PDF and
serving model are unchanged. Original run source bindings remain intact; exact
historical replay uses the archived source, as described in the study guide.

## Paper presentation

The eight-page paper is readable, and its main qualifications are already in
the abstract, results and conclusion. Keep the primary 4k comparison prominent;
the larger-run findings are supporting evidence. Preserve the separation of
selection, calibration and test, and the explicit warning about test reuse.

For the next manuscript revision, replace "quantifies reading removed" in
Section 4.2 with: "The skipped fraction measures documents removed from the
review queue; reading time and economic consequences were not measured."
Place the four-step decision rule from the study guide beside the methods and
state the exact data/model access procedure in Code and reproducibility.
These are editorial proposals; this branch does not create a new paper edition
or relabel existing results as independently confirmed.

The task-specific related-work references were checked against the
[Laya calibration study](https://arxiv.org/abs/2609.33843), the
[financial complaint benchmark](https://github.com/stuaninauts/bench-laya-vs-llms)
and [Borchers' report](https://www.drborchers.com/en/blog/do-you-need-a-decision-model/).
They support positioning this as a domain-specific empirical comparison.
This was a focused reference check, not an exhaustive novelty search.

## Verification scope

Both independent CPU verifiers passed: 16 main comparisons and three
development comparisons. The active metrics matched all 16 main runs on
selection, calibration and test; calibration policies matched their saved
decisions. Policy selection also matched the original implementation on 300
tie-heavy/single-class cases. Ruff and the lockfile check passed. API startup and
classical research inference succeeded in the Python 3.13 environment with no
PyTorch or collection packages. Regenerated manuscript fragments and figures
matched the published files byte for byte; all eight PDF pages were inspected.
A small synthetic CPU fit exercised selection, calibration, timing and resume
without a test split; it is an implementation check, not research evidence.
The existing PDF was not recompiled. Snyk reported the same 57 findings on the
base and branch, with no new finding fingerprints; this is not a clean security
bill for the existing repository. Most are CLI path-handling findings, alongside
command-execution, archive-extraction and URL-check findings requiring separate
contextual triage.

No source collection, paid labeling, research-model retraining or new test inference was
performed. Fresh empirical validation and access to private reproduction inputs
remain open work; a code refactor cannot resolve them.
