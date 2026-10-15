# Binary triage paper: editorial edition

[Read the paper](paper.pdf). This edition tightens the introduction, methods and
limitations, removes repeated qualifications, and keeps acquisition detail in
the supporting evidence. Results, tables, figures, references and conclusions
retain the original study's meaning. The later FinBERT 25k development experiment
is separate and is not a new test comparison.

The methods description also corrects an overgeneralization in the original prose:
Laya averages logits from overlapping windows; FinBERT averages embeddings from
non-overlapping source windows before classification. This follows `encoders.py`;
the implementation and all recorded results are unchanged.

The [original publication](../paper-binary-v2/README.md) and
[scientific evidence](../reports/binary-v2/README.md) remain byte-identical.
This directory versions the prose revision without rewriting those releases.
`paper-manifest.json` records the parent edition, frozen evidence and new artifacts;
`review.json` records visual and reproduction checks for the revised PDF.
The paper has six main pages, one reference page and one supporting appendix page.
Prose fragments in `generated/` are editorial copies; the original renderer and
its unchanged generated outputs remain in the preceding publication.

## Build

Extract [arxiv-source.zip](arxiv-source.zip) into an empty directory and run:

```sh
tectonic --keep-logs paper.tex
```

Tectonic 0.17.0 is the tested engine. The source archive contains only LaTeX and
two figures. To rebuild the release from this checkout:

```sh
python decision_models/build_paper.py --engine tectonic
```

The builder checks the original release's hashes and unchanged result numbers,
tables, figures and references, then compiles a fresh source archive. Compilation
does not replace visual review; inspect rendered pages after editing.
