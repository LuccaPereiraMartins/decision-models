"""Write a tiny synthetic binary-v2-shaped split directory for smoke-testing fit_tabular.py.

The documents are templated, labels are synthetic and nothing here is research evidence.
It exists so the full feature -> tabular model -> calibration path can run on a laptop
without the private frozen corpus.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

from schemas import BINARY_LABELS, BINARY_TASK, body_hash, save


ROUTINE = (
    "The company filed its quarterly dividend notice in the ordinary course of business.",
    "The registrant announces the date of its annual shareholder meeting.",
    "An exhibit is furnished containing the investor presentation previously posted online.",
    "The board approved routine amendments to the committee charters.",
)
REVIEW = (
    "The company entered a definitive merger agreement with a strategic acquirer.",
    "The auditor resigned and the registrant disclosed a material weakness in internal controls.",
    "The issuer received a going concern qualification and suspended guidance.",
    "The chief executive officer departed with immediate effect amid an investigation.",
)
FILLER = "Forward-looking statements involve risks and uncertainties described in periodic reports."


def documents(count, seed):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        label = BINARY_LABELS[int(rng.random() < 0.45)]
        core = rng.choice(REVIEW if label == "review_worthy" else ROUTINE)
        noise = rng.choice(ROUTINE + REVIEW) if rng.random() < 0.15 else ""
        body = f"Document {seed}-{index}. {core} {noise} " + " ".join([FILLER] * rng.randint(1, 6))
        rows.append(
            {
                "date": f"2026-0{1 + index % 9}-15",
                "body": body,
                "llm": {"label": label, "uncertainty": 0.1},
                "human": {"label": None, "uncertainty": None},
                "source": "sec",
                "document_type": "8-K",
            }
        )
    return rows


def write(output, train=600, selection=200, calibration=120):
    output.mkdir(parents=True, exist_ok=True)
    splits = {}
    for offset, (name, count) in enumerate((("train", train), ("selection", selection), ("calibration", calibration))):
        content = "".join(json.dumps(row) + "\n" for row in documents(count, offset))
        (output / f"{name}.jsonl").write_bytes(content.encode())
        splits[name] = {"count": count, "sha256": hashlib.sha256(content.encode()).hexdigest()}
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    sizes = sorted({min(n, train) for n in (train // 4, train // 2, train)})
    save({"subsets": {str(n): [body_hash(r) for r in rows[:n]] for n in sizes}}, output / "subsets.json")
    manifest = {"task_id": BINARY_TASK, "labels": list(BINARY_LABELS), "synthetic": True, "splits": splits}
    manifest["subsets_sha256"] = hashlib.sha256((output / "subsets.json").read_bytes()).hexdigest()
    save(manifest, output / "manifest.json")
    return sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=600)
    args = parser.parse_args()
    print(f"Synthetic subsets {write(args.output, args.train)} in {args.output}")
