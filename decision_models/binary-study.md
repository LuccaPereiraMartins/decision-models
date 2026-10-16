# Bounded binary financial-document triage study

The bounded study has completed all 16 frozen comparisons, including both 16k
LoRA fits. Read [the paper](paper-binary-v2/paper.pdf), especially Section 5 and
Tables 2–4, and [the replay instructions](reports/binary-v2/README.md).
[Execution checkpoints](binary-checkpoint.md) retain the progress record.
The protocol below records the agreed scope and safeguards.

This is a new, versioned study following the user-approved binary redesign. The
three-class benchmark, trained artifacts, frozen results and published packages
remain historical evidence. The new task has exactly `routine` and `review_worthy`;
uncertainty remains metadata. Judge source content alone. Clearly administrative
content can be routine; an unresolved potentially significant financial development
goes to review. Labels measure agreement with Luna, not analyst-validated materiality.

The question is whether Laya and its adaptation improve financial-document triage
relative to a finance encoder and a lexical classifier, at equal training data and
with explicit missed-document, analyst-workload and compute tradeoffs.

## Data and labels

- Keep 25,000 training documents and nested 4k/16k/25k subsets. Add eligible ECB
  training-period releases through deterministic replacements of SEC training groups.
- Retain existing evaluation memberships and top up to exactly 1,000 selection,
  500 calibration and 2,000 test documents. Dates remain training through March 2026,
  development April–June and test July–September. Related groups cannot cross splits.
- Luna Flex re-decides prior unclear references from the body alone. Existing definite
  references are retained with their original provenance; 200 training references get
  a blinded binary consistency audit, which does not silently override them.
- New references use a pinned binary rubric. All original annotations remain available.
  Split selection is independent of labels and model scores.
- ECB expansion follows available eligible institutional/statistical releases without
  quotas. The completed acquisition adds 534 releases: 509 training-period, 12 development
  and 13 test-period. These are acquisition counts; duplicate/group filtering can reduce
  benchmark support. See [the source audit](binary-ecb-extension.md).
- The durable Luna ledger has a separate **$2 incremental cap**, including ambiguous
  reservations. Restarting cannot reset the allowance or use released historical
  reservations to enlarge it. Retain $0.50 of the user's stated balance in reserve.

The revised test deliberately reuses the prior 1,800 bodies. Disclose reuse and reference
changes; do not call it a fresh independent confirmation or compare binary macro F1
directly with the previous three-class score. Freeze all choices before its new evaluation.

## Fixed matrix

| System | Task fitting |
| --- | --- |
| Laya native SDK | None |
| Laya head | 4k |
| Laya LoRA | 4k; 16k if the guarded runtime fits |
| Frozen FinBERT/logistic | 4k, 16k, 25k |
| FinBERT LoRA | 4k, 16k |
| Word TF-IDF/logistic | 4k, 16k, 25k |
| Majority | Count shared training labels; no optimization |

All trained core systems use identical 4k bodies. Primary comparison: Laya LoRA versus
word TF-IDF/logistic at 4k. The larger runs answer sample-efficiency questions. Preserve
matched epoch-zero controls as supporting evidence; SDK pooling and adapted pooling differ.

Use pinned pretrained weights, seed 42, unweighted document-level cross-entropy, up to
three epochs and raw selection macro F1 checkpoint selection, with epoch zero eligible.
Laya head uses learning rate 0.00003; LoRA uses head/encoder rates 0.0001, rank eight,
alpha 16 and dropout 0.05. Laya context is 1,024; FinBERT is 512. Uniform whole-document
pooling keeps all normalized source tokens. Logistic regularization and class weighting
use the existing bounded selection grid. Frozen feature caches retain recipe/body/source
checksums; recomputation is acceptable if a source change invalidates an old cache.

Run GPU jobs serially by default, with independent CPU work overlapping where useful.
Resource timing must not overlap competing workloads. Do not launch the old full matrix,
25k LoRA, other encoders, generative LLMs, extra seeds/context sweeps or RL/RLCD.

Native SDK inference caps each GPU pass at eight windows while retaining its normal
whole-document windowing and most-confident aggregation. The uncapped development
attempt triggered GPU-to-CPU fallback and was preserved/excluded before evaluation.
Binary SDK calls reject observed CPU fallback, so GPU resource measurements cannot
silently include it. This batching correction changes memory use, not the selected
cohorts, source coverage or model-selection rule.

## Evaluation and paper

Primary endpoint: two-class macro F1. Also report review precision/recall, false negatives
with their review-reference denominator, 2×2 confusion matrices, and skipped/reviewed
documents. Temperature and empirical 1%/5% miss-target policies fit only calibration;
10%/20%/30% skip-budget thresholds preserve whole ties. Report achieved test workloads,
not assumed matching or population guarantees. Binary Brier uses the sum over both
class columns (twice the positive-class squared error).

Use 2,000 paired whole-group bootstrap replicates, with intervals conditional on the
single fitted run. Report source/length support and shared errors without causal or
multiplicity-adjusted claims. Measure training time, warm mean/P95 latency and memory
on a common sample; record source/class composition and optimizer steps on learning curves.

Target 6–8 main pages excluding references and appendix. Lead with operational system
comparisons, adaptation and sample efficiency. Keep binary confusion and learning curves;
move full configurations, provenance, calibration derivations, resource detail and error
slices to organized supporting evidence. Deliver LaTeX, compiled PDF, source archive and
a separate inspectable binary package with CPU replay. No model weights/source bodies
belong in that public evidence package. Visually inspect every PDF page.

## Schedule, controls and handoff

The goal started **13 October 2026, 21:27:30 UTC**. Stop training and model evaluation
by **16 October, 09:27:30 UTC (10:27:30 London)**; finish handoff by
**16 October, 21:27:30 UTC (22:27:30 London)**. The ignored session receipt under
`training_runs/research-binary-v2/session.json` binds deadlines and original artifact hashes.

Complete the seven-system core before extensions. Prioritize FinBERT LoRA 16k, then
Laya LoRA 16k. Forecast the latter from measured 4k runtime and cohort volume; admit it
only if a 30% margin plus two hours for controls/evaluation fit before the cutoff.
Preserve partial checkpoints and state limitations honestly; stopped fits are not results.

Provide reviews every 12 hours and continue ordinary authorized work without approval
requests. The heartbeat `review-bounded-binary-triage-study` supplies those touchpoints.
Verify live PID, creation time and command before intervention or restarting a worker.

Deliver three draft PRs: data/protocol, bounded experiments, paper/evidence. Push owned
commits periodically and verify origin/current-head CI/PR metadata. Keep existing PRs
open/draft and leave `technical-primer.md` untouched. Complete only after exact split
counts, reference/group identity, finite two-class outputs, metric/policy arithmetic,
resource evidence, replay, PDF inspection and remote publication are verified.

Implementation entrypoints are `prepare_binary.py`, `label.py --task binary-v2`, and
`study_binary.py`. The bounded runner never calls the old full-grid fit method.

`binary_pipeline.py` continues the existing approved label controller, labels the
newly selected documents, freezes data and starts this bounded runner. It records
live process identities and checks source/preparation hashes; do not duplicate it
or edit its bound runtime sources while it is running.

After `models_complete_report_pending`, run `report_binary.py`. It checks frozen
artifacts, ordered cohorts, finite zero/best/final checkpoints and selected epochs,
exports a separate body-free package, and requires `binary_replay.py` to pass before
generating manuscript results. The draft source is in `paper-binary-v2/`; generated
LaTeX has been compiled and every page and figure visually inspected. The public
package regenerates the source and figures without model runtime or network;
two fresh archive builds match all nine rendered pages. Exact evidence and paper
hashes accompany the publication receipts. The old manuscript and core package
remain separate historical evidence.
