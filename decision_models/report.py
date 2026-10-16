"""Regenerate published tables and figures from independently verified binary evidence."""

import argparse
import gzip
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def name(run):
    c = run["configuration"]
    if c["family"] == "research_sdk":
        return "Laya native SDK"
    if c["family"] == "research_linear":
        return {"majority": "Training majority", "tfidf": "Word TF-IDF/logistic", "finbert": "Frozen FinBERT/logistic"}[
            c["kind"]
        ]
    return {"laya": "Laya", "finbert": "FinBERT"}[c["kind"]] + " " + {"head": "head", "lora": "LoRA"}[c["adaptation"]]


def number(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "--" if value is None else f"{100 * value:.1f}\\%"


def interval(value, digits=3):
    return f"$[{number(value['lower'], digits)}, {number(value['upper'], digits)}]$"


def table(caption, label, headers, rows, columns=None):
    columns = columns or ("X" + "r" * (len(headers) - 1))
    content = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        f"\\begin{{tabularx}}{{\\linewidth}}{{@{{}}{columns}@{{}}}}",
        "\\toprule",
        " & ".join(headers) + "\\\\",
        "\\midrule",
    ]
    content.extend(" & ".join(map(str, row)) + "\\\\" for row in rows)
    return "\n".join([*content, "\\bottomrule", "\\end{tabularx}", "\\end{table}"])


def plot(bundle, output, freeze):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 220,
        }
    )
    runs, statistics = bundle["runs"], bundle["bootstrap"]["models"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), layout="constrained")
    maximum = max(
        np.asarray(runs[key]["test"]["metrics"]["overall"]["confusion_matrix"]).max()
        for key in (freeze["primary_linear"], freeze["primary_decision"])
    )
    for axis, key in zip(axes, (freeze["primary_linear"], freeze["primary_decision"]), strict=True):
        matrix = np.asarray(runs[key]["test"]["metrics"]["overall"]["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for i in range(2):
            for j in range(2):
                axis.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > maximum / 2 else "black",
                    fontsize=12,
                )
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Routine", "Review"],
            yticklabels=["Routine", "Review"],
            xlabel="Prediction",
            ylabel="Teacher reference",
            title=name(runs[key]),
        )
    fig.savefig(output / "confusion.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), layout="constrained")
    groups = {}
    for key, run in runs.items():
        config = run["configuration"]
        eligible = (config["family"] == "research_linear" and config.get("kind") in ("tfidf", "finbert")) or (
            config["family"] == "research_neural" and config.get("adaptation") == "lora"
        )
        if run["entry"]["role"] in ("core", "curve") and eligible:
            groups.setdefault(name(run), []).append(key)
    colors = ["#244b74", "#b45625", "#416b43", "#654678"]
    styles = ["o", "s", "^", "D"]
    for (title, keys), color, marker in zip(groups.items(), colors, styles):
        keys.sort(key=lambda key: runs[key]["configuration"]["samples"])
        x = np.array([runs[key]["configuration"]["samples"] / 1000 for key in keys])
        for axis, metric in zip(axes, ("macro_f1", "dangerous_miss_rate"), strict=True):
            score_key = "false_negative_rate" if metric == "dangerous_miss_rate" else metric
            y = np.array([runs[key]["test"]["metrics"]["overall"][score_key] for key in keys])
            lows = np.array([statistics[key][metric]["lower"] for key in keys])
            highs = np.array([statistics[key][metric]["upper"] for key in keys])
            axis.plot(x, y, color=color, marker=marker, label=title)
            axis.vlines(x, lows, highs, color=color, alpha=0.6, linewidth=1)
    for axis, title in zip(axes, ("Macro F1", "Review-reference miss rate"), strict=True):
        axis.set(xlabel="Training documents (thousands)", ylabel=title, xticks=[4, 16, 25])
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    fig.savefig(output / "learning-curves.png")
    plt.close(fig)


def build(output, paper):
    subprocess.run([sys.executable, str(output / "verify.py"), str(output)], check=True)
    bundle = json.loads(gzip.decompress((output / "evidence.json.gz").read_bytes()))
    freeze = read(output / "freeze.json")
    runs, statistics = bundle["runs"], bundle["bootstrap"]["models"]
    core = [key for key, run in runs.items() if run["entry"]["role"] in ("core", "reference")]
    order = {
        title: i
        for i, title in enumerate(
            (
                "Training majority",
                "Word TF-IDF/logistic",
                "Frozen FinBERT/logistic",
                "FinBERT LoRA",
                "Laya native SDK",
                "Laya head",
                "Laya LoRA",
            )
        )
    }
    core.sort(key=lambda key: order[name(runs[key])])
    generated, figures = paper / "generated", paper / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)

    def save_tex(filename, content):
        (generated / filename).write_bytes((content.rstrip() + "\n").encode())

    counts = bundle["benchmark_manifest"]["splits"]
    save_tex(
        "data-table.tex",
        table(
            "Frozen cohort support. The test retains 1,800 earlier bodies; reference labels are revised.",
            "tab:data",
            ["Cohort", "Documents", "Routine", "Review", "Groups", "SEC", "ECB"],
            [
                (
                    key.title(),
                    counts[key]["count"],
                    counts[key]["labels"].get("routine", 0),
                    counts[key]["labels"].get("review_worthy", 0),
                    len({row["group_id"] for row in bundle["cohorts"][key]}),
                    counts[key]["sources"].get("sec", 0),
                    counts[key]["sources"].get("ecb", 0),
                )
                for key in counts
            ],
        ),
    )
    support = runs[freeze["primary_decision"]]["test"]["metrics"]["overall"]["per_class"]["review_worthy"]["support"]
    rows, policies = [], []
    for key in core:
        run, score = runs[key], runs[key]["test"]["metrics"]["overall"]
        timing = run["performance"]["latency_seconds"]
        rows.append(
            (
                name(run),
                f"{number(score['macro_f1'])} {interval(statistics[key]['macro_f1'])}",
                f"{score['false_negatives']}/{support}",
                percent(score["skip_fraction"]),
                number(1000 * timing["mean"], 2),
                number(1000 * timing["p95"], 2),
            )
        )
        miss, workload = [score["policies"][policy] for policy in ("0.01", "skip_0.2")]
        policies.append(
            (
                name(run),
                f"{miss['dangerous_misses']}/{support}",
                percent(miss["workload_reduction"]),
                f"{workload['dangerous_misses']}/{support}",
                percent(workload["workload_reduction"]),
            )
        )
    tables = [
        table(
            "Core systems on the revised 2,000-document evaluation. Default decisions; false negatives (FN) are divided by review-reference support. "
            "All trainable systems use the same 4k bodies; majority counts those labels, and SDK has no task fit. Warm inference includes full document processing.",
            "tab:main",
            ["System", "F1 (95\\% CI)", "FN", "Skipped", "Mean ms", "P95 ms"],
            rows,
        ),
        table(
            "Achieved test outcomes for two calibration policies. The left pair uses an empirical 1\\% calibration miss target; "
            "the right pair maximizes calibration skips within 20\\%. All policies skip only routine predictions and preserve ties. "
            "Achieved test workloads are not assumed equal.",
            "tab:policy",
            ["System", "1\\% miss: FN", "Skipped", "20\\% skip: FN", "Skipped"],
            policies,
        ),
    ]
    contrasts = bundle["bootstrap"]["contrasts"]
    increments = []
    for key, contrast in contrasts.items():
        if key.endswith("sample-increment"):
            left, right = [runs[contrast[k]] for k in ("left", "right")]
            delta = left["test"]["metrics"]["overall"]["macro_f1"] - right["test"]["metrics"]["overall"]["macro_f1"]
            increments.append(
                (
                    name(left),
                    f"{right['configuration']['samples'] // 1000}k to {left['configuration']['samples'] // 1000}k",
                    number(delta, 4),
                    interval(contrast["metrics"]["macro_f1"], 4),
                )
            )
    tables.append(
        table(
            "Paired training-size F1 increments for fixed system families. Conditional 95\\% intervals; supplementary comparisons are uncorrected.",
            "tab:increments",
            ["System", "Training change", "$\\Delta$ F1", "95\\% interval"],
            increments,
            "XXrr",
        )
    )
    main_path = generated / "main-results.tex"
    main = main_path.read_text(encoding="utf-8")
    for content in tables:
        label = re.search(r"\\label\{([^}]+)\}", content)[1]
        pattern = r"\\begin\{table\}.*?\\end\{table\}"
        matches = [block for block in re.findall(pattern, main, re.S) if "\\label{" + label + "}" in block]
        if len(matches) != 1:
            raise ValueError(f"Expected one existing result table: {label}")
        main = main.replace(matches[0], content)
    save_tex("main-results.tex", main)

    supporting = []
    controls, recipes, resources, prior_attempts = [], [], [], []
    for key, run in runs.items():
        c = run["configuration"]
        if run["entry"]["role"] == "control":
            continue
        size = c.get("samples", 0)
        size_label = "Count 4k" if c.get("kind") == "majority" else f"{size // 1000}k" if size else "None"
        if run["entry"].get("matched_zero"):
            control = runs[run["entry"]["matched_zero"]]
            contrast = contrasts[key + "-adaptation"]["metrics"]["macro_f1"]
            controls.append(
                (
                    name(run),
                    f"{size // 1000}k",
                    number(control["test"]["metrics"]["overall"]["macro_f1"]),
                    number(run["test"]["metrics"]["overall"]["macro_f1"]),
                    interval(contrast),
                )
            )
        context = c.get("max_len", c.get("context", {}).get("max_len", 512 if c.get("kind") == "finbert" else "--"))
        if c["family"] == "research_neural":
            recipes.append(
                (name(run), size_label, str(context), str(c["trainable_parameters"]), str(run["history"]["best_epoch"]))
            )
        timing = run["performance"]
        controller = run.get("controller", {})
        resources.append(
            (
                name(run),
                size_label,
                number(controller.get("successful_attempt_seconds"), 1),
                number(run.get("history", {}).get("seconds", c.get("fit_seconds")), 1),
                number(timing["training_peak_vram_bytes"] / 2**30, 2) if "training_peak_vram_bytes" in timing else "--",
                number(timing.get("peak_vram_bytes", 0) / 2**30, 2),
            )
        )
        if controller.get("prior_attempt_seconds", 0) > 0:
            prior_attempts.append(
                f"{name(run)} also used {number(controller['prior_attempt_seconds'], 1)} seconds in prior interrupted attempts"
            )
    supporting.append(
        table(
            "Matched pre-update comparisons. The initial adapted pooling/head state is the control; it is distinct from native SDK inference. "
            "Larger fixed recipes reuse a matched setup's pre-update control. Intervals are supplementary.",
            "tab:controls",
            ["System", "Train", "Zero F1", "Fit F1", "$\\Delta$ F1 interval"],
            controls,
        )
    )
    supporting.append(
        table(
            "Completed neural configurations. All use seed 42 and at most three epochs; epoch zero is eligible. "
            "A context limit includes task/model overhead, rather than denote full source coverage. All normalized source tokens are windowed.",
            "tab:recipes",
            ["System", "Train", "Context", "Trainable", "Selected epoch"],
            recipes,
        )
    )
    supporting.append(
        table(
            "Recorded resources. Command time includes preparation, fitting and development evaluation. "
            "Fit phase is the neural history or linear preparation/search duration, including selection overhead. "
            "Fit GiB records neural-fit peaks only; '--' denotes an unavailable measurement. "
            "Inference peaks can include resident fitting state. Frozen feature/cache costs are included in command time and itemized in the evidence.",
            "tab:resources",
            ["System", "Train", "Command s", "Fit phase s", "Fit GiB", "Infer GiB"],
            resources,
        )
    )
    if prior_attempts:
        supporting.append(
            "\\FloatBarrier\n"
            + "; ".join(prior_attempts)
            + ". These attempts are excluded from the successful-command column; cumulative elapsed time and attempt receipts remain in the evidence package."
        )
    save_tex("supporting.tex", "\n\n".join(supporting))
    plot(bundle, figures, freeze)
    return {
        "comparisons": len(runs),
        "core_systems": len(core),
        "status": "Tables and figures reproduced; prose unchanged",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=HERE / "reports/binary-v2")
    parser.add_argument("--paper", type=Path, default=HERE / "paper")
    args = parser.parse_args()
    print(json.dumps(build(args.evidence.resolve(), args.paper.resolve()), indent=2))
