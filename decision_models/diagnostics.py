"""Post-test descriptive diagnostics; never fit models or select discard thresholds."""
from collections import Counter

import numpy as np

from evaluate_study import bootstrap, interval
from protocol import metrics
from schemas import LABELS


def class_f1(matrix):
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(-1) + matrix.sum(-2)
    return np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator > 0)


def review_routing(matrix):
    """Routine discards; both review-worthy and unclear escalate, without changing predictions."""
    result = np.zeros((*matrix.shape[:-2], 2, 2), dtype=matrix.dtype)
    result[..., 0, 0] = matrix[..., 0, 0]
    result[..., 0, 1] = matrix[..., 0, 1:].sum(-1)
    result[..., 1, 0] = matrix[..., 1:, 0].sum(-1)
    result[..., 1, 1] = matrix[..., 1:, 1:].sum((-2, -1))
    return result


def analyze(predictions, inputs, statistics):
    primary, reference = inputs["primary"], inputs["reference"]
    names = (primary, reference)
    rows = predictions[primary]
    identities = [(r["body_sha256"], r["group_id"], r["reference"]) for r in rows]
    if any([(r["body_sha256"], r["group_id"], r["reference"]) for r in predictions[n]] != identities for n in names):
        raise ValueError("Diagnostics require identical ordered cohorts")
    truth = np.array([LABELS.index(r["reference"]) for r in rows])
    predicted = {n: np.array([LABELS.index(r["prediction"]) for r in predictions[n]]) for n in names}
    groups, group_index = np.unique([r["group_id"] for r in rows], return_inverse=True)
    draws = np.random.default_rng(42).multinomial(len(groups), [1 / len(groups)] * len(groups), size=2000)
    point, sampled, routing, routed_samples = {}, {}, {}, {}
    for name in names:
        counts = np.bincount(group_index * 9 + truth * 3 + predicted[name], minlength=len(groups) * 9)
        counts = counts.reshape(len(groups), 9)
        point[name] = counts.sum(0).reshape(3, 3)
        sampled[name] = (draws @ counts).reshape(2000, 3, 3)
        binary = review_routing(point[name])
        support = int(binary[1].sum())
        routing[name] = {"macro_f1": float(class_f1(binary).mean()), "confusion_matrix": binary.tolist(),
                         "review_required_support": support, "review_required_misses": int(binary[1, 0]),
                         "review_required_miss_rate": float(binary[1, 0] / support),
                         "review_required_recall": float(binary[1, 1] / support),
                         "critical_review_to_routine": int(point[name][1, 0]),
                         "unclear_to_routine": int(point[name][2, 0]),
                         "workload_reduction": float(binary[:, 0].sum() / len(rows))}
        binary_draws = review_routing(sampled[name])
        denominator = binary_draws[:, 1].sum(1)
        routed_samples[name] = {
            "macro_f1": class_f1(binary_draws).mean(1),
            "review_required_miss_rate": np.divide(binary_draws[:, 1, 0], denominator,
                                                   out=np.full(2000, np.nan), where=denominator > 0),
            "workload_reduction": binary_draws[:, :, 0].sum(1) / binary_draws.sum((1, 2)),
        }
        routing[name]["intervals"] = {key: interval(value) for key, value in routed_samples[name].items()}
        # The new calculations must reproduce the original three-class endpoint exactly.
        saved = statistics["models"][name]["macro_f1"]
        replay = interval(class_f1(sampled[name]).mean(1))
        np.testing.assert_allclose([replay["lower"], replay["upper"]], [saved["lower"], saved["upper"]], atol=1e-12)
    contribution = (class_f1(point[primary]) - class_f1(point[reference])) / 3
    class_difference = (class_f1(sampled[primary]) - class_f1(sampled[reference])) / 3
    decomposition = [{"label": label, "primary_f1": float(class_f1(point[primary])[i]),
                      "reference_f1": float(class_f1(point[reference])[i]), "macro_f1_contribution": float(contribution[i]),
                      "interval": interval(class_difference[:, i])} for i, label in enumerate(LABELS)]
    miss = {name: (truth == 1) & (predicted[name] == 0) for name in names}
    overlap = {"both": int((miss[primary] & miss[reference]).sum()),
               "primary_only": int((miss[primary] & ~miss[reference]).sum()),
               "reference_only": int((miss[reference] & ~miss[primary]).sum()),
               "neither": int(((truth == 1) & ~miss[primary] & ~miss[reference]).sum()),
               "review_worthy_support": int((truth == 1).sum()),
               "prediction_agreement": float((predicted[primary] == predicted[reference]).mean())}

    def slice_result(mask, title):
        subset = {name: [r for r, take in zip(predictions[name], mask, strict=True) if take] for name in names}
        if not len(subset[primary]):
            return {"slice": title, "count": 0}
        scores = {name: metrics(value) for name, value in subset.items()}
        labels = dict(Counter(r["reference"] for r in subset[primary]))
        paired = bootstrap(subset, reference)["models"][primary]["paired_delta_vs_reference"]
        return {"slice": title, "count": len(subset[primary]),
                "groups": len({r["group_id"] for r in subset[primary]}), "label_support": labels,
                "fixed_three_label_f1_ceiling": len(labels) / 3,
                "models": scores, "macro_f1_delta": scores[primary]["macro_f1"] - scores[reference]["macro_f1"],
                "paired_intervals": paired,
                "critical_miss_overlap": {"both": int((miss[primary] & miss[reference] & mask).sum()),
                                          "primary_only": int((miss[primary] & ~miss[reference] & mask).sum()),
                                          "reference_only": int((miss[reference] & ~miss[primary] & mask).sum())}}

    family = [slice_result(np.array([r["family"] == key for r in rows]), key)
              for key in sorted({r["family"] for r in rows})]
    cuts = np.quantile(inputs["selection_characters"], [.25, .5, .75])
    bands = np.searchsorted(cuts, [r["characters"] for r in rows], side="right")
    boundaries = [0, *cuts, None]
    length = []
    for index in range(4):
        title = f"{boundaries[index]:,.0f} to <{boundaries[index + 1]:,.0f}" if index < 3 else f">={cuts[-1]:,.0f}"
        length.append(slice_result(bands == index, title))
    policies = []
    for name in inputs["policy_models"]:
        for key, value in metrics(predictions[name])["policies"].items():
            policies.append({"model": name, "calibration_target": key, **value,
                             "workload_interval": statistics["models"][name]["workload_" + key],
                             "miss_rate_interval": statistics["models"][name]["dangerous_miss_rate_" + key]})
    return {"status": "Exploratory diagnostics added after seeing primary held-out results; no model or policy selection",
            "primary": primary, "reference": reference, "documents": len(rows), "groups": len(groups),
            "methods": {"repeats": 2000, "seed": 42, "routing_labels": ["routine", "requires_review"],
                        "length_cutpoints_characters": cuts.tolist(), "length_definition": "Selection-set quartiles; upper boundary enters next band",
                        "slice_intervals": "Paired whole-group resampling within each slice; three fixed labels even when absent",
                        "limits": "Conditional on one trained seed; no multiplicity adjustment; source and length are confounded"},
            "class_decomposition": decomposition,
            "macro_f1_delta": float(contribution.sum()), "routing": routing,
            "paired_routing_intervals": {key: interval(value - routed_samples[reference][key])
                                         for key, value in routed_samples[primary].items()},
            "critical_miss_overlap": overlap, "family_slices": family, "length_slices": length,
            "frozen_policies": policies}


def figures(output, result, labels):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True)
    for ax, key, title in zip(axes, ("family_slices", "length_slices"), ("Source families", "Selection-defined character-length bands"), strict=True):
        slices = [r for r in result[key] if r["count"]]
        for i, row in enumerate(slices):
            bounds = row["paired_intervals"]["macro_f1"]
            ax.hlines(i, bounds["lower"], bounds["upper"], color="#1855a0")
            ax.plot(row["macro_f1_delta"], i, "o", color="#1855a0")
        ax.set(yticks=range(len(slices)), yticklabels=[f"{r['slice']} (n={r['count']})" for r in slices],
               xlabel="Laya minus TF-IDF macro F1", title=title)
        ax.axvline(0, color="gray", linestyle="--", linewidth=.8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
        ax.tick_params(labelsize=8)
    fig.suptitle("Exploratory slices; paired within-slice group intervals", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "diagnostic-slices.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    models = list(dict.fromkeys(r["model"] for r in result["frozen_policies"]))
    colors = {name: plt.get_cmap("tab10")(i) for i, name in enumerate(models)}
    for row in result["frozen_policies"]:
        x, y = row["workload_reduction"] * 100, row["dangerous_miss_rate"] * 100
        wx, wy = row["workload_interval"], row["miss_rate_interval"]
        ax.hlines(y, wx["lower"] * 100, wx["upper"] * 100, color=colors[row["model"]], alpha=.6)
        ax.vlines(x, wy["lower"] * 100, wy["upper"] * 100, color=colors[row["model"]], alpha=.6)
        ax.scatter(x, y, color=colors[row["model"]], marker="o" if row["calibration_target"] == "0.01" else "s")
    for target in (1, 5):
        ax.axhline(target, linestyle="--", color="gray", linewidth=.8)
    handles = [Line2D([], [], color=colors[n], label=labels[n]) for n in models]
    handles += [Line2D([], [], color="black", marker=m, linestyle="none", label=f"{t}% calibration target")
                for m, t in (("o", 1), ("s", 5))]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set(xlabel="Test workload removed (%)", ylabel="Teacher review-worthy miss rate (%)",
           title="Existing frozen policies; marginal group-bootstrap 95% intervals", ylim=(0, None))
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(output / "diagnostic-policies.png", dpi=180)
    plt.close(fig)
