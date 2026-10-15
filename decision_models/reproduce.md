# Reproducing the research

There are three distinct operations: replaying recorded arithmetic, compiling the
paper, and rerunning models. Only the first two are self-contained public releases.

## Replay the published evidence

From the repository root, with Python and NumPy installed:

```sh
python decision_models/reports/binary-v2/verify.py
python decision_models/reports/binary-finbert25k/verify.py
```

The first verifies all 16 frozen binary comparisons, including metrics, policies,
fitting choices, timing and paired bootstrap intervals. The second checks the
separate FinBERT 4k/16k/25k development comparison and GPU observations. Neither
command trains a model, performs model inference or calls an API.

Each package binds its files with SHA-256 hashes and supplies ordered references,
predictions, configurations and original source snapshots. CPU replay validates
recorded evidence; it cannot validate the teacher's financial judgment.

## Compile the paper

Extract [the editorial source archive](paper-binary-revised/arxiv-source.zip) into
an empty directory and run `tectonic --keep-logs paper.tex`. Tectonic 0.17.0 is the
tested engine. The archive contains LaTeX and figures; no model runtime is needed.

The [editorial edition](paper-binary-revised/README.md) uses the original frozen
results. The [original paper](paper-binary-v2/README.md), source archive and
evidence package remain byte-identical. To regenerate its tables and figures
from saved predictions, follow the [evidence-package instructions](reports/binary-v2/README.md).

## Rerun fitting or inference

Exact model reproduction additionally requires the local frozen document bodies,
ordered subsets, checkpoints, pinned pretrained weights and recorded dependencies.
The public evidence excludes bodies and weights. A fresh download can differ from
the retained historical text; public URLs alone cannot reconstruct exact inputs.
Acquisition and attribution records are in [sources.md](sources.md).

Restore the source snapshot and dependency versions recorded by the relevant run
into an isolated checkout. Verify the manifest, ordered subsets and artifact hashes
before replay. Preserve the original directory layout when configurations contain
absolute paths. Do not edit frozen metadata or delete holdout ownership markers.

For a new experiment, use a separate protocol and output directory. Selection and
calibration are fitting cohorts. The revised test was already inspected, and the
25k extension was chosen afterwards; new generalization claims require fresh
held-out data. The [binary protocol](binary-study.md) records completed recipes and
the distinction between the core comparison and learning curves.

The [historical reproduction guide](reproduce-core-v1.md) applies only to the earlier
three-class study and its preserved core package.
