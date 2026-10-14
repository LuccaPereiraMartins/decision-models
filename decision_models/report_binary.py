"""Generate concise manuscript results from the independently replayed binary study."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from export_binary import export, read, write
from protocol import fingerprint
from schemas import HERE


def name(run):
    c = run["configuration"]
    if c["family"] == "research_sdk":
        return "Laya native SDK"
    if c["family"] == "research_linear":
        return {"majority": "Training majority", "tfidf": "Word TF-IDF/logistic",
                "finbert": "Frozen FinBERT/logistic"}[c["kind"]]
    return {"laya": "Laya", "finbert": "FinBERT"}[c["kind"]] + " " + {"head": "head", "lora": "LoRA"}[c["adaptation"]]


def number(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "--" if value is None else f"{100 * value:.1f}\\%"


def interval(value, digits=3):
    return f"$[{number(value['lower'], digits)}, {number(value['upper'], digits)}]$"


def table(caption, label, headers, rows, columns=None):
    columns = columns or ("X" + "r" * (len(headers) - 1))
    content = ["\\begin{table}[!htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
               "\\small", "\\setlength{\\tabcolsep}{3.5pt}",
               f"\\begin{{tabularx}}{{\\linewidth}}{{@{{}}{columns}@{{}}}}", "\\toprule",
               " & ".join(headers) + "\\\\", "\\midrule"]
    content.extend(" & ".join(map(str, row)) + "\\\\" for row in rows)
    return "\n".join([*content, "\\bottomrule", "\\end{tabularx}", "\\end{table}"])


def plot(bundle, output, freeze):
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 220})
    runs, statistics = bundle["runs"], bundle["bootstrap"]["models"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), layout="constrained")
    for axis, key in zip(axes, (freeze["primary_linear"], freeze["primary_decision"]), strict=True):
        matrix = np.asarray(runs[key]["test"]["metrics"]["overall"]["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                          color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=12)
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Routine", "Review"], yticklabels=["Routine", "Review"],
                 xlabel="Prediction", ylabel="Teacher reference", title=name(runs[key]))
    fig.savefig(output / "confusion.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), layout="constrained")
    groups = {}
    for key, run in runs.items():
        config = run["configuration"]
        eligible = ((config["family"] == "research_linear" and config.get("kind") in ("tfidf", "finbert")) or
                    (config["family"] == "research_neural" and config.get("adaptation") == "lora"))
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
            axis.vlines(x, lows, highs, color=color, alpha=.6, linewidth=1)
    for axis, title in zip(axes, ("Macro F1", "Review-reference miss rate"), strict=True):
        axis.set(xlabel="Training documents (thousands)", ylabel=title, xticks=[4, 16, 25])
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(loc="best", fontsize=8)
    fig.savefig(output / "learning-curves.png")
    plt.close(fig)


def build(splits, study, output, paper):
    export(splits, study, output)
    bundle = json.loads(gzip.decompress((output / "evidence.json.gz").read_bytes()))
    freeze = read(output / "freeze.json")
    runs, statistics = bundle["runs"], bundle["bootstrap"]["models"]
    core = [key for key, run in runs.items() if run["entry"]["role"] in ("core", "reference")]
    if len(core) != 7:
        raise ValueError("Main comparison must contain exactly seven core systems")
    main_order = {title: i for i, title in enumerate(("Training majority", "Word TF-IDF/logistic",
                 "Frozen FinBERT/logistic", "FinBERT LoRA", "Laya native SDK", "Laya head", "Laya LoRA"))}
    core.sort(key=lambda key: main_order[name(runs[key])])
    generated, figures = paper / "generated", paper / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)

    def save_tex(filename, text):
        (generated / filename).write_bytes((text.rstrip() + "\n").encode())

    primary, reference = freeze["primary_decision"], freeze["primary_linear"]
    p, r = [runs[key]["test"]["metrics"]["overall"] for key in (primary, reference)]
    delta = statistics[primary]["paired_delta_vs_reference"]["macro_f1"]
    advantage = ("does not resolve an F1 difference" if delta["lower"] <= 0 <= delta["upper"] else
                 "is positive for this fitted comparison" if delta["lower"] > 0 else "is negative for this fitted comparison")
    latencies = [runs[key]["performance"]["latency_seconds"]["mean"] for key in (primary, reference)]
    ratio = latencies[0] / latencies[1]
    support = p["per_class"]["review_worthy"]["support"]
    summary = (f"At 4k, Laya LoRA achieves macro F1 {number(p['macro_f1'],4)} versus {number(r['macro_f1'],4)} "
               f"for word TF-IDF/logistic. The paired F1 difference interval {interval(delta,4)} {advantage}. "
               f"They miss {p['false_negatives']} and {r['false_negatives']} of {support} review-worthy references, "
               f"while skipping {percent(p['skip_fraction'])} and {percent(r['skip_fraction'])} of all documents. "
               f"Laya's warm mean latency is {ratio:.1f} times the lexical classifier's on the recorded machine.")
    save_tex("abstract.tex", "Financial document triage must trade less analyst reading against missed information and processing cost. "
             "We compare seven decision-model, finance-encoder and lexical systems under a binary source-only teacher convention. "
             "The study uses 25,000 training documents, separate 1,000/500 selection/calibration cohorts and a revised 2,000-document evaluation. "
             "Trainable core systems share 4,000 documents. " + summary + " "
             "Larger training subsets separately measure sample efficiency, and calibration-selected skip policies expose workload/miss tradeoffs. "
             "The test reuses previously inspected documents; these results measure teacher agreement, not independent confirmation or analyst-validated materiality. "
             "We provide body-free evidence and independent CPU replay.")
    counts = bundle["benchmark_manifest"]["splits"]
    save_tex("data-table.tex", table("Frozen cohort support. The test retains 1,800 earlier bodies; reference labels are revised.",
             "tab:data", ["Cohort", "Documents", "Routine", "Review", "Groups", "ECB"],
             [(key.title(), counts[key]["count"], counts[key]["labels"].get("routine",0), counts[key]["labels"].get("review_worthy",0),
               len({row["group_id"] for row in bundle["cohorts"][key]}), counts[key]["sources"].get("ecb",0)) for key in counts]))
    rows, policy_rows = [], []
    for key in core:
        run, score = runs[key], runs[key]["test"]["metrics"]["overall"]
        timing = run["performance"]["latency_seconds"]
        rows.append((name(run), f"{number(score['macro_f1'])} {interval(statistics[key]['macro_f1'])}",
                     f"{score['false_negatives']}/{support}", percent(score["skip_fraction"]),
                     number(1000*timing["mean"],2), number(1000*timing["p95"],2)))
        miss, workload = [score["policies"][policy] for policy in ("0.01", "skip_0.2")]
        policy_rows.append((name(run), f"{miss['dangerous_misses']}/{support}", percent(miss["workload_reduction"]),
                            f"{workload['dangerous_misses']}/{support}", percent(workload["workload_reduction"])))
    main = ["\\subsection{Equal-data system comparison}", summary,
            table("Core systems on the revised 2,000-document evaluation. Default decisions; false negatives (FN) are divided by review-reference support. "
                  "All trainable systems use the same 4k bodies; majority counts those labels, and SDK has no task fit. Warm inference includes full document processing.",
                  "tab:main", ["System", "F1 (95\\% CI)", "FN", "Skipped", "Mean ms", "P95 ms"], rows),
            "\\begin{figure}[tbp]\\centering\n\\includegraphics[width=\\linewidth]{confusion.png}\n"
            "\\caption{Default two-class confusion matrices for the fixed primary pair. Rows are teacher references; columns are model predictions. Counts sum to 2,000 in each matrix.}"
            "\\label{fig:confusion}\\end{figure}",
            "\\subsection{Review workload and missed references}",
            "Table~\\ref{tab:policy} applies calibration-selected thresholds unchanged to test. "
            "The 1\\% fitting target and 20\\% skip budget produce different achieved workloads across systems; neither is a test guarantee. "
            "Default and thresholded results should be compared jointly with the number of references missed.",
            table("Achieved test outcomes for two calibration policies. The left pair uses an empirical 1\\% calibration miss target; "
                  "the right pair maximizes calibration skips within 20\\%. All policies skip only routine predictions and preserve ties. "
                  "Achieved test workloads are not assumed equal.", "tab:policy",
                  ["System", "1\\% FN", "Skipped", "20\\% FN", "Skipped"], policy_rows),
            "\\subsection{Adaptation and sample efficiency}",
            "Figure~\\ref{fig:learning} separates training size from the shared-4k comparison. "
            "Intervals are conditional on fixed fitted runs and are not corrected for multiple comparisons. "
            "The larger run also uses more optimization steps; it does not isolate sample count from every other fitting effect.",
            "\\begin{figure}[tbp]\\centering\n\\includegraphics[width=\\linewidth]{learning-curves.png}\n"
            "\\caption{Nested training-size curves for completed recipes. Left: macro F1. Right: review-reference miss rate at default decisions. "
            "Vertical segments are marginal whole-group 95\\% intervals. LoRA curves stop at the largest completed authorized fit.}"
            "\\label{fig:learning}\\end{figure}"]
    contrasts = bundle["bootstrap"]["contrasts"]
    adapted = []
    for key in core:
        matched = runs[key]["entry"].get("matched_zero")
        if matched:
            zero = runs[matched]["test"]["metrics"]["overall"]["macro_f1"]
            fitted = runs[key]["test"]["metrics"]["overall"]["macro_f1"]
            adapted.append(f"{name(runs[key])}: {number(fitted-zero,4)} "
                           + interval(contrasts[key+"-adaptation"]["metrics"]["macro_f1"],4))
    if adapted:
        main.append("The 4k F1 changes relative to matched pre-update states are " + "; ".join(adapted)
                    + ". These compare each complete fitting recipe with its own initial state; they do not isolate pretraining or equalize classifier budgets. "
                    "Selected epochs and control scores appear in the supporting tables.")
    if "descriptive" in bundle:
        overlap = bundle["descriptive"]["primary_miss_overlap"]
        main.append(f"The primary systems share {len(overlap['both'])} default missed references, "
                    f"with {len(overlap['laya_only'])} additional Laya-only and {len(overlap['tfidf_only'])} TF-IDF-only misses. "
                    "This overlap describes the fixed predictions; it does not evaluate a new combined routing rule.")
    increments = []
    for key, contrast in contrasts.items():
        if key.endswith("sample-increment"):
            left, right = [runs[contrast[k]] for k in ("left", "right")]
            value = left["test"]["metrics"]["overall"]["macro_f1"] - right["test"]["metrics"]["overall"]["macro_f1"]
            increments.append((name(left), f"{right['configuration']['samples']//1000}k to {left['configuration']['samples']//1000}k",
                               number(value,4), interval(contrast["metrics"]["macro_f1"],4)))
    main.append(table("Paired training-size F1 increments for fixed system families. Conditional 95\\% intervals; supplementary comparisons are uncorrected.",
                      "tab:increments", ["System", "Training change", "$\\Delta$ F1", "95\\% interval"], increments, "XXrr"))
    gate = bundle.get("schedule", {})
    if gate.get("status") in ("deferred", "infeasible", "stopped"):
        main.append("The 16k Laya LoRA fit is absent from the frozen comparison because its bounded runtime gate "
                    + {"deferred":"did not admit the forecast within the deadline", "infeasible":"recorded GPU memory infeasibility",
                       "stopped":"stopped computation before a complete fit"}[gate["status"]] + ". Partial checkpoints are not reported as completed results.")
    save_tex("main-results.tex", "\n\n".join(main))
    audit = bundle["benchmark_manifest"]["retained_label_consistency"]
    save_tex("discussion.tex", f"The primary pair's observed F1 interval {advantage}, with a measured {ratio:.1f}-fold mean-latency ratio. "
             "Neither F1 nor processing cost alone establishes an operating choice: default misses and reading removed differ, "
             "and calibration policies must be assessed at their achieved evaluation workload. "
             f"The training-only reference audit agrees on {percent(audit['agreement'])} of 200 balanced retained labels. "
             "That consistency result cannot substitute for expert reference validity or estimate a population annotation error rate.")
    save_tex("conclusion.tex", summary + " "
             "These conditional comparisons identify concrete accuracy/workload/resource tradeoffs under the teacher's binary convention. "
             "The next evidential step is expert reference validation and a fresh representative evaluation cohort, with methods fixed before evaluation.")
    save_tex("reproducibility.tex", "The command \\texttt{python decision\\_models/reports/binary-v2/verify.py} "
             "replays the published metrics, policies, timing summaries and paired intervals on CPU without API calls. "
             "Code and evidence are available in the \\href{https://github.com/LuccaPereiraMartins/systematic-trading}{study repository}. "
             f"The revised freeze identity is \\texttt{{{bundle['freeze_sha256'][:12]}}}; its full hash and source identities accompany the package.")
    supporting = []
    controls, recipes, resources = [], [], []
    for key, run in runs.items():
        c = run["configuration"]
        if run["entry"]["role"] == "control":
            continue
        size = c.get("samples",0)
        size_label = "Count 4k" if c.get("kind") == "majority" else f"{size//1000}k" if size else "None"
        if run["entry"].get("matched_zero"):
            control = runs[run["entry"]["matched_zero"]]
            contrast = contrasts[key+"-adaptation"]["metrics"]["macro_f1"]
            controls.append((name(run), f"{size//1000}k", number(control["test"]["metrics"]["overall"]["macro_f1"]),
                             number(run["test"]["metrics"]["overall"]["macro_f1"]), interval(contrast)))
        context = c.get("max_len", c.get("context",{}).get("max_len",512 if c.get("kind") == "finbert" else "--"))
        recipes.append((name(run), size_label, str(context),
                        str(c.get("trainable_parameters","--")), str(run.get("history",{}).get("best_epoch","--"))))
        timing=run["performance"]
        resources.append((name(run), size_label, number(run.get("controller",{}).get("elapsed_seconds"),1),
                          number(run.get("history",{}).get("seconds",c.get("fit_seconds")),1),
                          number(timing.get("training_peak_vram_bytes",0)/2**30,2), number(timing.get("peak_vram_bytes",0)/2**30,2)))
    supporting.append(table("Matched pre-update comparisons. The initial adapted pooling/head state is the control; it is distinct from native SDK inference. "
                            "Larger fixed recipes reuse a matched setup's pre-update control. Intervals are supplementary.",
                            "tab:controls", ["System", "Train", "Zero F1", "Fit F1", "$\\Delta$ F1 interval"], controls))
    supporting.append(table("Completed configurations. All neural recipes use seed 42 and at most three epochs; epoch zero is eligible. "
                            "A context limit includes task/model overhead, rather than denote full source coverage. All normalized source tokens are windowed.",
                            "tab:recipes", ["System", "Train", "Context", "Trainable", "Epoch"], recipes))
    supporting.append(table("Recorded resources. Command time includes preparation, fitting and development evaluation; "
                            "fit time is the neural history duration or linear fit measurement. GPU columns use peak allocated GiB at each recorded stage; "
                            "resident fitting state/caches can contribute to the inference-stage peak. "
                            "Cached feature computation and optimizer-step details are retained in the public package.",
                            "tab:resources", ["System", "Train", "Command s", "Fit s", "Train GiB", "Infer GiB"], resources))
    source_rows=[]
    for source in sorted({row["source"] for row in bundle["cohorts"]["test"]}):
        slices=[runs[key]["test"]["metrics"]["by_source"][source] for key in (reference,primary)]
        source_rows.append((source.upper(),slices[0]["count"],slices[0]["per_class"]["review_worthy"]["support"],
                            number(slices[0]["macro_f1"]),number(slices[1]["macro_f1"]),slices[0]["false_negatives"],slices[1]["false_negatives"]))
    supporting.append(table("Descriptive source support for the fixed primary pair. Both class labels stay fixed within slices; "
                            "missing-class support limits attainable macro F1. Source and length are confounded, and small-source results do not establish generalization.",
                            "tab:sources", ["Source", "$n$", "Review", "TF F1", "Laya F1", "TF FN", "Laya FN"], source_rows))
    save_tex("supporting.tex", "\n\n".join(supporting))
    plot(bundle, figures, freeze)
    write(output / "manuscript-inputs.json", {"status": "generated_from_cpu_replayed_results; PDF review pending",
          "freeze_sha256": bundle["freeze_sha256"], "evidence_sha256": fingerprint(output / "evidence.json.gz"),
          "renderer_sha256": fingerprint(Path(__file__)),
          "generated_files": {str(path.relative_to(paper)):fingerprint(path) for path in sorted(paper.rglob("*"))
                              if path.is_file() and path.suffix in (".tex", ".png")}})
    return {"core_systems":len(core),"completed_comparisons":len(runs),"status":"manuscript source generated; PDF checks pending"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits",type=Path,default=HERE/"data/research/binary-v2")
    parser.add_argument("--study",type=Path,default=HERE/"training_runs/research-binary-v2/study")
    parser.add_argument("--output",type=Path,default=HERE/"reports/binary-v2")
    parser.add_argument("--paper",type=Path,default=HERE/"paper-binary-v2")
    args=parser.parse_args()
    print(json.dumps(build(args.splits,args.study,args.output,args.paper),indent=2))
