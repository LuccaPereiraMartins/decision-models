# Critical review for academic ML readers

Reviewed 23 October 2026 against `master` at `be606f2`, including the completed
4k/16k/25k study. The contribution is an empirical comparison of financial-document
routing, adaptation and processing cost. Scores measure agreement with the Luna
reference rubric; analyst utility and investment usefulness require separate evidence.

## Findings and their limits

1. **The equal-data comparison shows an operational tradeoff.** At 4k documents,
   Laya LoRA scores macro F1 .7843 and TF-IDF/logistic .7645. The paired difference
   interval [-.0022, .0423] establishes neither superiority nor equivalence. Laya
   skips 648 documents and misses 268/1,543 review references; TF-IDF skips 492 and
   misses 188. Laya's measured mean latency is 64.9 times higher. Interpret F1,
   misses, skipped documents and processing cost together (paper Table 2).

2. **Both LoRA learning curves now reach 25k.** These runs were planned but
   completed after the initial report because of runtime. Their recipes were fixed
   without using earlier test results, and the same evaluation protocol applies at
   every size. Laya improves from .8424 at 16k to .8503 at 25k, with 171 to 165
   misses at unchanged 28.0% skipped documents; its paired F1 increment is .0079
   [-.0039, .0204]. FinBERT improves from .8140 to .8288, with 202 to 193 misses
   and 28.8% to 29.0% skipped; its increment is .0148 [.0019, .0284]. Command time
   rises from 20.66 to 37.17 hours for Laya and 2.51 to 4.36 hours for FinBERT.
   These measurements complete the planned scaling analysis (Figure 2, Table 4).

3. **Similar selection F1 does not mean identical fitted models.** Laya 25k scores
   higher at epochs one and two, then reaches .86159 at epoch three versus .86164
   at 16k. Their final classes differ on 56/1,000 shared selection documents.
   The public replay checks both histories and those prediction differences.
   Cross-family comparisons also differ in context, pooling, parameter counts,
   objectives and optimization budgets. Larger nested subsets change source mix
   and optimizer steps. Bootstrap intervals condition on one fitted run and leave
   training variability and multiplicity unresolved.

4. **Reference validity bounds the research claim.** Training and evaluation use
   the same teacher, combining retained definite labels with new binary decisions.
   The 93% consistency result is a balanced 200-document training audit, not analyst
   agreement or a prevalence-weighted annotation error rate. A blinded analyst
   sample would test whether the rubric captures useful review decisions. This is
   future validation of operational utility, rather than a prerequisite for
   reporting the teacher-agreement benchmark accurately.

5. **The cohort limits transfer claims across all model sizes.** The 2,000 test
   documents include 1,800 previously inspected bodies and 1,870 SEC documents.
   The 1,967 event/near-duplicate groups support the recorded paired bootstrap,
   but grouping cannot prove independence across issuers or recurring templates.
   All comparisons share the same test cohort. Fresh documents and a representative
   source mix are needed for external generalization evidence.

6. **Workload is a document-count proxy.** Skipped fractions do not measure reading
   minutes, difficulty or economic loss. Under the 1% calibration miss target, the
   4k TF-IDF and Laya systems miss 20 and 7 test references while skipping different
   fractions. Both 25k LoRA systems miss 9/1,543, with 6.2% skipped for FinBERT and
   7.1% for Laya (Table 3). Calibration has 365 review references, so a 1% empirical
   target permits only three observed misses. It supplies no population guarantee.

## Reading and reproducing the implementation

Follow the study guide's path through `baselines.make`, `fit_baselines.fit`,
`protocol.py`, the neural window/loss code, and frozen evaluation. The refactor
keeps classical inference independent of PyTorch and collection packages, moves
sampling helpers into `schemas.py`, scores each policy threshold once, and checks
classifier hashes before deserialization. A trusted manifest is still required.

The three public evidence packages cover the original 16 comparisons, FinBERT's
25k development records, and both 25k test comparisons. Their CPU verifiers replay
metrics, decisions, fitting records, provenance and paired intervals without source
bodies or neural weights. The latter package completes the same study while
preserving the original package identities.

Exact model inference and fitting additionally need retained text snapshots,
selected weights, archived producer sources and recorded package versions. The
public TF-IDF artifact supports inference on supplied documents. `prepare.py`
extends historical cohorts, so collecting today's pages alone cannot rebuild the
frozen benchmark. An artifact submission should specify an access route and reuse
conditions for retained inputs/weights, plus a standalone fresh-data example.

The locked Python 3.13 environment matches the CPU serving artifact's NumPy/SciPy
versions. The README separately documents the recorded Windows CUDA override,
Torch `2.14.0+cu126`, and the need to retain it with `uv run --no-sync`.

## Paper review priorities

Start with the abstract, results, learning curves and conclusion. Keep the equal-4k
comparison and the completed 4k/16k/25k analysis together: one controls training
size, the other measures scaling under the fixed recipes. All sizes have the same
reference-label and cohort limitations. Preserve selection/calibration/test roles.

For the next editorial pass, replace "quantifies reading removed" in Section 4.2
with "measures documents removed from the review queue; reading time and economic
consequences were not measured." Put the four-step decision rule from the guide
beside the methods and state the data/model access procedure explicitly. The paper's
evidence link is pinned to an immutable commit so merged-branch deletion is safe.

Related-work pointers include the [Laya calibration study](https://arxiv.org/abs/2609.33843),
[financial complaint benchmark](https://github.com/stuaninauts/bench-laya-vs-llms),
and [Borchers report](https://www.drborchers.com/en/blog/do-you-need-a-decision-model/).
They position the work as a financial-domain empirical comparison; these pointers
are not an exhaustive novelty assessment.

## Verification scope

All three independent CPU replays pass. Active refactored metrics agree with 50
saved cohort results: selection, calibration and test for the original 16 models,
plus both 25k tests. Calibration policies match the saved original decisions; the
policy refactor also matches 300 tied-probability and single-class cases. The
minimal pinned CPU environment loads the serving artifact and classical research
runtime without PyTorch or collection packages. Ruff and the lockfile check pass.
The CUDA wheel command resolves in a dry run; the installed original runtime
reports Torch `2.14.0+cu126`, CUDA 12.6 and an available GPU. No clean-environment
CUDA installation or new neural fit was performed.

The original evidence packages and serving artifact retain their published hashes.
The paper changes only its evidence hyperlink; all eight page renders and the
independent source-archive rebuild match the inspected paper pixel for pixel.
No new collection, paid labeling, research training or test inference was needed.
