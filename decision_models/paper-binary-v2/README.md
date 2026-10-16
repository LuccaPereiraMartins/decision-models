# Binary financial-document triage paper

[Read the compiled paper](paper.pdf). Start with Section 5 and Tables 2-4 for the
core comparison, missed-reference/workload policies and training-size findings.
The paper is nine pages: main text through page 7, references, and one supporting
appendix page. It includes FinBERT and Laya LoRA at both 4k and 16k.

The fixed 4k comparison gives Laya LoRA macro F1 0.7843 versus word TF-IDF/logistic
0.7645, with paired difference interval [-0.0022, 0.0423]. Laya skips more reading
but misses more teacher-review references and has 64.9 times the mean warm latency.
These conditional findings concern a SEC-dominated teacher-labelled benchmark
that reuses earlier test bodies, rather than expert materiality or fresh confirmation.

## Rebuild

Extract [arxiv-source.zip](arxiv-source.zip) into a fresh directory and run:

```sh
tectonic --keep-logs paper.tex
```

The tested engine is Tectonic 0.17.0. The archive contains only the main source,
references, generated LaTeX fragments and two figures. It requires standard LaTeX
packages and no shell escape, executables, document bodies or model weights.
All nine pages and both figures were visually inspected. Two independent fresh
archive builds match pixel-for-pixel at 150 dpi, and public evidence regenerates
all source fragments/figures without model runtime or network access.

[paper-manifest.json](paper-manifest.json) binds the PDF, source members, archive,
build environment and evidence hashes. [The evidence package](../reports/binary-v2/README.md)
provides independent CPU replay and table/figure regeneration instructions.

Authors: Lucca Pereira Martins and Matthew Ball. Corresponding email:
lp573@cantab.ac.uk. No affiliations or second email have been inferred.
The prior `paper/` and `reports/core-v1/` packages remain preserved.
