"""Build a source-backed Markdown/PDF research report from verified frozen test artifacts."""

from collections import Counter, defaultdict
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
from xml.sax.saxutils import escape

import numpy as np

from evaluate_study import bootstrap, validate_freeze
from prepare import family
from protocol import breakdown, fingerprint
from schemas import HERE, LABELS, annotation, body_hash, load_split, read_records, save


def number(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def ci(value):
    return "-" if value["lower"] is None else f"[{value['lower']:.3f}, {value['upper']:.3f}]"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def label(config):
    if config.get("kind") == "majority":
        return "Majority class"
    kind = {"laya": "Laya", "tfidf": "Word TF-IDF", "char": "Char TF-IDF", "combined": "Combined TF-IDF",
            "majority": "Majority", "finbert": "FinBERT", "bge": "BGE-small", "modernbert": "ModernBERT-large",
            "qwen17": "Qwen3-1.7B", "qwen4": "Qwen3-4B"}[config.get("kind", "laya")]
    suffix = config.get("classifier", config.get("adaptation", "SDK"))
    suffix = "LoRA" if suffix == "lora" else suffix
    if config["family"] == "research_zero":
        suffix += " epoch zero"
    return kind + " " + suffix


def human_audit(path, records, predictions, reference):
    if path is None:
        return {"status": "Pending blinded non-expert review", "resolved": 0, "unresolved": 0}
    reviewed = read_records(path)
    if {body_hash(r) for r in reviewed} != {body_hash(r) for r in records} or len(reviewed) != len(records):
        raise ValueError("Human audit must overlay exactly the frozen test bodies")
    labels = {body_hash(r): r["human"]["label"] for r in reviewed if r["human"]["label"] is not None}
    unresolved = sum(bool(r["human"].get("unresolved")) for r in reviewed)
    subset = {name: [{**r, "reference": labels[r["body_sha256"]], "reference_source": "human"}
                     for r in rows if r["body_sha256"] in labels] for name, rows in predictions.items()}
    truth = {body_hash(r): r["llm"]["label"] for r in records}
    return {"status": "Blinded non-expert audit; stratified review sample, not analyst ground truth",
            "source_sha256": fingerprint(path), "resolved": len(labels), "unresolved": unresolved,
            "teacher_agreement": sum(truth[k] == value for k, value in labels.items()) / len(labels) if labels else None,
            "models": {name: breakdown(rows) for name, rows in subset.items()},
            "bootstrap": bootstrap(subset, reference) if labels else None}


def corpus_tables(prepared):
    parts = []
    if prepared.get("census"):
        parts += [table(["Input", "Raw documents", "Family / provider", "Document dates"],
                        [[Path(row["input"]).name + (" (legacy)" if row["legacy"] else ""), row["rows"],
                          ", ".join(row["families"]) + " / " + ", ".join(row["providers"]),
                          f"{row['start']} to {row['end']}"] for row in prepared["census"]]),
                  "Raw acquisition counts precede duplicate quarantine, sampling and labeling. Legacy inputs were previously "
                  "experimented on and are eligible only for training. SEC dates are filing dates, including release exhibits, "
                  "and may be later than publication elsewhere. Other sources use original article dates or publication metadata; "
                  "temporal splits use these recorded document dates at day precision."]
    if prepared.get("legacy_extraction_check"):
        legacy = prepared["legacy_extraction_check"]
        parts += [table(["Legacy text formatter", "Primary filings reconstructed", "Exact old-body matches"],
                        [[f"edgartools {legacy['sdk_version']}", legacy["attempted"], legacy["matched_documents"]]]),
                  "Offline reconstruction of retained primary HTML links exact old-body hashes across text-extraction formats. "
                  "These links also quarantine related release exhibits from fresh evaluation. Matching candidates are not "
                  "additional exclusion counts; their groups enter the exclusion rules below. Canonical model inputs are unchanged."]
    if prepared.get("legacy_event_check"):
        legacy = prepared["legacy_event_check"]
        parts += [table(["Legacy event formatter", "Old bodies covered", "Verified events", "Matching source families"],
                        [[f"edgartools {legacy['sdk_version']}", legacy["legacy_bodies"], legacy["verified_events"],
                          ", ".join(f"{name}: {count}" for name, count in legacy["matched_families"].items())]]),
                  "Every old body is reproduced exactly from preserved submissions or retained index/primary responses. "
                  "The resulting filing-event links also quarantine release-only candidates whose parent filing is absent "
                  "from the new sample. Matching source rows enter the group exclusion rules and are not additional "
                  "exclusion counts. Evidence hashes and reconstruction methods are recorded with the corpus."]
    if prepared.get("exclusions_by_family"):
        parts += [table(["Family", "Exclusion reason", "Documents"],
                        [[name, reason.replace("_", " "), count]
                         for name, counts in prepared["exclusions_by_family"].items() for reason, count in counts.items()])]
    if prepared.get("partitions"):
        parts += [table(["Partition", "Available after exclusions", "Selected before labeling"],
                        [[name, counts["available"], counts["chosen"]] for name, counts in prepared["partitions"].items()]),
                  "Selection applies the recorded split ceilings and train/validation/test 8-K share cap; these counts are distinct "
                  "from labeled support below."]
    return parts


def training_composition(splits):
    manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
    digest = fingerprint(splits / "subsets.json")
    if digest != manifest["subsets_sha256"]:
        raise ValueError("Frozen training subsets changed")
    rows = load_split(splits, "train")
    by_hash = {body_hash(row): row for row in rows}
    subsets = json.loads((splits / "subsets.json").read_text(encoding="utf-8"))["subsets"]
    counts, previous = {}, []
    for size in sorted(subsets, key=int):
        identities = subsets[size]
        if len(identities) != int(size) or len(set(identities)) != int(size) or identities[:len(previous)] != previous:
            raise ValueError("Training composition requires distinct, nested frozen subsets")
        selected = [by_hash[key] for key in identities]
        counts[size] = {"documents": len(selected), "groups": len({row.get("group_id", body_hash(row)) for row in selected}),
                        "labels": dict(Counter(annotation(row)["label"] for row in selected)),
                        "families": dict(Counter(family(row) for row in selected)),
                        "providers": dict(Counter(row.get("source", "sec") for row in selected))}
        previous = identities
    return {"train_sha256": manifest["splits"]["train"]["sha256"], "subsets_sha256": digest, "subsets": counts}


def training_tables(composition):
    counts = composition["subsets"]
    parts = [table(["Training documents", "Independent groups", *LABELS],
                   [[size, value["groups"], *[value["labels"].get(name, 0) for name in LABELS]] for size, value in counts.items()])]
    for kind, title in (("families", "Family"), ("providers", "Provider")):
        names = sorted({name for value in counts.values() for name in value[kind]})
        parts.append(table([title, *counts], [[name, *[value[kind].get(name, 0) for value in counts.values()]] for name in names]))
    return parts


def uncertainty_bands(rows):
    """Fixed descriptive bands; these do not select a deployment threshold."""
    uncertainty = np.array([row["uncertainty"] for row in rows], dtype=float)
    if not np.isfinite(uncertainty).all() or ((uncertainty < 0) | (uncertainty > 1)).any():
        raise ValueError("Expected finite uncertainty in [0, 1]")
    bands = []
    bounds = (0.0, 0.1, 0.2, 0.4, 1.0)
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        mask = (uncertainty >= lower) & ((uncertainty <= upper) if upper == 1 else (uncertainty < upper))
        selected = [row for row, keep in zip(rows, mask, strict=True) if keep]
        support = sum(row["reference"] == "review_worthy" for row in selected)
        misses = sum(row["reference"] == "review_worthy" and row["prediction"] == "routine" for row in selected)
        bands.append({"lower": lower, "upper": upper, "count": len(selected),
                      "accuracy": sum(row["reference"] == row["prediction"] for row in selected) / len(selected) if selected else None,
                      "review_worthy_support": support, "dangerous_misses": misses,
                      "dangerous_miss_rate": misses / support if support else None})
    return bands


def uncertainty_tables(bands, configs, names):
    return [f"Calibrated uncertainty: {label(configs[name])}.\n\n" +
            table(["Uncertainty", "Documents", "Accuracy", "Review support", "Misses", "Miss rate"],
                  [[f"[{row['lower']:.2f}, {row['upper']:.2f}{']' if row['upper'] == 1 else ')'}",
                    row["count"], number(row["accuracy"]), row["review_worthy_support"],
                    row["dangerous_misses"], number(row["dangerous_miss_rate"])] for row in bands[name]])
            for name in names]


def reference_quote(records):
    """Local cost scenarios only: no API client, credentials, dispatch or paid benchmark."""
    import tiktoken
    from label import RUBRIC
    tokenizer = tiktoken.get_encoding("o200k_base")
    estimated = [len(tokenizer.encode(RUBRIC + r["body"], disallowed_special=())) + 128 for r in records]
    bound = [len((RUBRIC + r["body"]).encode("utf-8")) + 2048 for r in records]
    scenarios = {}
    for output in (128, 1024):
        scenarios[str(output)] = {"approximate_usd": sum(n / 1e6 * 1.25 * (2 if n > 272000 else 1) +
                                                         output / 1e6 * 5 * (1.5 if n > 272000 else 1) for n in estimated),
                                 "conservative_usd": sum(n / 1e6 * 1.25 * (2 if n > 272000 else 1) +
                                                         output / 1e6 * 5 * (1.5 if n > 272000 else 1) for n in bound)}
    return {"model": "gpt-6.1-sol", "tier": "flex", "executed": False, "documents": len(records),
            "rates_usd_per_million": {"input": 1.0, "cached_input": .05, "cache_write": 1.25, "output": 5.0},
            "pricing_checked": "2026-10-10", "pricing_source": "https://developers.openai.com/api/docs/pricing/",
            "tokenizer": "o200k_base approximation; UTF-8 byte upper bound plus 2048 prompt/schema overhead",
            "scenarios_output_tokens_per_document": scenarios,
            "limits": "Assumes all input written to cache, no reuse discounts. Above 272k input, 2x input/1.5x output. Reasoning counts as output; not a service guarantee."}


def cached_weights(config):
    if not config.get("model"):
        return None
    from huggingface_hub import try_to_load_from_cache
    from pathlib import Path
    configuration = try_to_load_from_cache(config["model"], "config.json", revision=config["revision"])
    if not isinstance(configuration, str):
        return None
    snapshot = Path(configuration).parent
    sizes = [p.stat().st_size for p in snapshot.rglob("*") if p.suffix in (".bin", ".pt", ".safetensors")]
    return sum(sizes) if sizes else None


def figures(output, entries, configs, results, intervals, main, primary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    grouped = defaultdict(list)
    for name, entry in entries.items():
        if entry["role"] == "curve":
            grouped[label(configs[name])].append(name)
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (title, names) in enumerate(grouped.items()):
        names.sort(key=lambda n: entries[n]["samples"])
        x = [entries[n]["samples"] for n in names]
        y = [results[n]["macro_f1"] for n in names]
        ax.plot(x, y, "o-", label=title, alpha=.8, color=plt.get_cmap("tab20")(index % 20))
    ax.set(xscale="log", xlabel="Training documents (validation-selected recipes)", ylabel="Test macro F1",
           ylim=(-.02, 1.05), title="Sample efficiency on the same temporal test cohort")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning-curves.png", dpi=180)
    plt.close(fig)
    names = [n for n, e in entries.items() if e["kind"] == "laya" and e["role"] in ("context", "curve") and
             e["samples"] == min(max(e["samples"] for e in entries.values()), 4000)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for adaptation in ("head", "lora"):
        selected = sorted([n for n in names if configs[n]["adaptation"] == adaptation], key=lambda n: configs[n]["max_len"])
        x, y = [configs[n]["max_len"] for n in selected], [results[n]["macro_f1"] for n in selected]
        if selected:
            line, = ax.plot(x, y, "o-", label="LoRA" if adaptation == "lora" else adaptation)
            ax.vlines(x, [intervals[n]["macro_f1"]["lower"] for n in selected],
                      [intervals[n]["macro_f1"]["upper"] for n in selected], color=line.get_color())
    ax.set(xlabel="Laya total context tokens", ylabel="Test macro F1", ylim=(-.02, 1.05),
           title="Context ablation; group-bootstrap 95% intervals")
    if names:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "context.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, name in enumerate(main):
        perf = json.loads((output.parent / entries[name]["directory"] / "performance.json").read_text())
        x = perf["batch_throughput"]["1"]["documents_per_second"]
        ax.scatter(x, results[name]["macro_f1"], color=plt.get_cmap("tab20")(index % 20),
                   label=f"{label(configs[name])} (n={entries[name]['samples']})")
    ax.set(xscale="log", xlabel="Warm single-document throughput (documents/s)", ylabel="Test macro F1", ylim=(-.02, 1.05),
           title="Agreement and local throughput; shared RTX 3060/CPU environment")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "throughput.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, primary, strict=True):
        matrix = np.array(results[name]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=["routine", "review", "unclear"],
               yticklabels=["routine", "review", "unclear"], xlabel="Prediction", ylabel="Teacher reference", title=label(configs[name]))
    fig.tight_layout()
    fig.savefig(output / "confusion.png", dpi=180)
    plt.close(fig)


def pdf(markdown, destination):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize, styles["BodyText"].leading = 10, 14
    styles["Heading2"].keepWithNext = True
    cell = ParagraphStyle("Cell", fontSize=7.5, leading=10, alignment=TA_LEFT)
    header = ParagraphStyle("Header", parent=cell, textColor=colors.white)
    width = A4[0] - 84
    flow = []
    def text(value):
        value = escape(value).replace("`", "")
        value = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#1855a0">\1</link>', value)
        return value
    lines, index = markdown.splitlines(), 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("| "):
            rows = []
            while index < len(lines) and lines[index].startswith("| "):
                columns = [value.strip() for value in lines[index].strip("| ").split("|")]
                if not all(value == "---" for value in columns):
                    rows.append([Paragraph(text(value), header if not rows else cell) for value in columns])
                index += 1
            widths = [width / len(rows[0])] * len(rows[0])
            if len(widths) > 4:
                widths = [width * .28] + [width * .72 / (len(widths) - 1)] * (len(widths) - 1)
            content = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            content.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#203b58")),
                                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")])]))
            flow += [KeepTogether([content, Spacer(1, 12)])] if len(rows) <= 6 else [content, Spacer(1, 12)]
            continue
        match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if match:
            image = Image(str(destination.parent / match[2]))
            image.drawHeight = image.imageHeight * width / image.imageWidth
            image.drawWidth = width
            flow += [image, Paragraph(text(match[1]), styles["Caption"] if "Caption" in styles else cell), Spacer(1, 12)]
        elif line:
            level = len(line) - len(line.lstrip("#"))
            style = styles["Title" if level == 1 else "Heading2" if level else "BodyText"]
            flow.append(Paragraph(text(line.lstrip("# ") if level else line), style))
        index += 1
    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(42, 25, "Financial-document triage | fixed-seed research study")
        canvas.drawRightString(A4[0] - 42, 25, str(document.page))
    SimpleDocTemplate(str(destination), title="Financial-document triage across public sources",
                      author="Financial-document triage research", pagesize=A4,
                      rightMargin=42, leftMargin=42, topMargin=38, bottomMargin=42).build(
        flow, onFirstPage=footer, onLaterPages=footer)


def exploratory_sections(result, labels, statistics):
    primary, reference = result["primary"], result["reference"]
    routing, overlap = result["routing"], result["critical_miss_overlap"]
    unclear = next(r for r in result["class_decomposition"] if r["label"] == "unclear")
    share = unclear["macro_f1_contribution"] / result["macro_f1_delta"]
    parts = ["", "## Exploratory diagnostics added after the primary results", "",
             "The following questions were added after inspecting the primary held-out results. They are descriptive, "
             "not prespecified hypotheses, new model choices or tuned discard rules. The original primary comparison "
             "and its conclusion remain unchanged. Intervals condition on one trained seed and have no multiplicity adjustment.",
             f"Approximately {100 * share:.1f}% of the observed macro-F1 point difference comes from the unclear class. "
             "Each class contributes its F1 difference divided by three; source-family F1 scores cannot be added this way. "
             "The contribution intervals all include zero. This decomposition is not a causal attribution or a resolved class-specific gain.",
             table(["Class", "Class F1 difference", "Contribution to macro-F1 difference", "Contribution 95% CI"],
                   [[r['label'], number(r['primary_f1'] - r['reference_f1'], 4), number(r['macro_f1_contribution'], 4), ci(r['interval'])]
                    for r in result['class_decomposition']]),
             "Both review-worthy and unclear predictions route to an analyst. The derived two-class view below collapses "
             "those existing actions, without changing a prediction. Its numerical scale and meaning differ from the original "
             "three-class endpoint; it does not replace it.",
             table(["Model", "Routing macro F1", "95% CI", "Review misses", "Review support", "Unclear discarded", "Workload removed"],
                   [[labels[n], number(routing[n]['macro_f1']), ci(routing[n]['intervals']['macro_f1']),
                     routing[n]['review_required_misses'], routing[n]['review_required_support'], routing[n]['unclear_to_routine'],
                     number(routing[n]['workload_reduction'])] for n in (primary, reference)]),
             f"Paired routing-F1 difference interval: {ci(result['paired_routing_intervals']['macro_f1'])}. "
             f"Laya routes {round((routing[reference]['workload_reduction'] - routing[primary]['workload_reduction']) * result['documents'])} "
             "additional documents for review in this cohort. Its lower point count of required-review misses therefore "
             "comes with additional workload; neither a routing-F1 nor a miss-rate advantage is resolved by these intervals.",
             "### Shared and model-specific critical misses",
             table(["Critical-miss pattern", "Documents"], [["Both models", overlap['both']],
                   ["Laya only", overlap['primary_only']], ["TF-IDF only", overlap['reference_only']],
                   ["Neither model", overlap['neither']]]),
             f"These four cells partition {overlap['review_worthy_support']} teacher review-worthy documents. Overall "
             f"three-class prediction agreement is {100 * overlap['prediction_agreement']:.2f}%. Complementary errors "
             "motivate a future ensemble/cascade study, but these counts omit its false-positive, workload and compute "
             "tradeoffs. They do not measure a new deployed policy. Any such policy needs validation selection and fresh test data. "
             "Common misses cannot distinguish student errors from questionable teacher references without independent review.",
             "### Source and document-length differences",
             "![Primary-model differences within source and length slices](diagnostic-slices.png)",
             "Dots are Laya minus TF-IDF point differences; bars are paired 95% intervals from 2,000 whole-group "
             "resamples within each slice. Length bands use character-count quartiles from the 837-document selection "
             "cohort, chosen mechanically for this post-hoc analysis. Their cutpoints are " +
             ", ".join(f"{v:,.0f}" for v in result['methods']['length_cutpoints_characters']) + " characters. "
             "Test bands need not have equal counts. Character lengths are not model token counts. Source and length "
             "overlap, so these slices do not isolate a causal effect of context or architecture.",
             table(["Family", "Documents", "Routine", "Review-worthy", "Unclear", "Fixed-label F1 ceiling"],
                   [[r['slice'], r['count'], *[r['label_support'].get(k, 0) for k in LABELS], number(r['fixed_three_label_f1_ceiling'])]
                    for r in result['family_slices']]),
             "All slice scores retain the three declared labels. A slice without unclear references has a maximum "
             "possible macro F1 of 2/3 under this definition, not one. Compare models within the same slice; raw scores "
             "across families have different label support and attainable ceilings. The 6-K slice has the largest positive "
             "point difference; its nominal interval excludes zero. This exploratory, uncorrected comparison does not "
             "establish universal source superiority. All four length-band intervals include zero, including the longest band.",
             "### Uncertainty around the frozen discard rules",
             "![Workload and critical misses for the existing frozen policies](diagnostic-policies.png)",
             "Circles/squares identify the original 1%/5% calibration targets. Dashed lines show those targets, not "
             "deployment guarantees. Horizontal and vertical bars are separate marginal 95% group intervals, not joint "
             "confidence regions. These points and intervals reuse the original frozen policies and statistics; no "
             "threshold sweep, ensemble or model selection was performed."]
    paired = statistics['models'][primary]['paired_delta_vs_reference']
    def percent_ci(value):
        return f"[{100 * value['lower']:.2f}, {100 * value['upper']:.2f}]"
    parts += [table(["Calibration target", "Workload-difference 95% CI (pp)", "Critical-miss-rate difference 95% CI (pp)"],
                    [[key, percent_ci(paired['workload_' + key]), percent_ci(paired['dangerous_miss_rate_' + key])]
                     for key in ('0.01', '0.05')]),
              "Differences are Laya minus TF-IDF. More workload removed is favorable; a higher critical-miss rate is "
              "unfavorable. At the 1% target, Laya removes less workload and the miss-rate difference is unresolved. "
              "At the 5% target, its nominal interval indicates more teacher review-worthy misses while the workload "
              "difference remains unresolved. These conditional, uncorrected comparisons help interpret the existing "
              "systems; they are not permission to choose a new operating rule from this test set."]
    return parts


def development_sections(result):
    models = result['models']
    rows = [models[str(n)] for n in (4000, 16000)]
    parts = ['', '## Development-only FinBERT sample-efficiency extension', '',
             'After the original held-out results were inspected, the user authorized one additional fixed-recipe '
             'FinBERT LoRA fit at 16,000 training documents. This section uses train, selection and calibration only. '
             'It is not a 65th held-out comparison and does not revise the frozen primary result. A new held-out '
             'comparison requires fresh data. Selection scores choose the checkpoint and are optimistic fitting '
             'evidence; calibration scores also use fitting data. No significance or generalization claim follows.',
             'Both sizes start from the same pinned FinBERT weights and identical initial task-head parameters. '
             'The recipe stays fixed: context 512, rank 8, alpha 16, dropout 0.05, unweighted cross-entropy, '
             'learning rate 0.0001 for head and adapters, seed 42 and three epochs. Epoch zero remains eligible; '
             'both sizes select epoch three. Pooling, source coverage and the 837-selection/363-calibration cohorts '
             'are unchanged. There is no new learning-rate or seed sweep.',
             table(['Training docs', 'Selection F1', 'Critical misses / 552', 'Unclear discarded / 93', 'Default discards / 837', 'Calibration F1'],
                   [[r['training_documents'], number(r['selection']['macro_f1'], 4), r['selection']['default_discard']['dangerous_misses'],
                     r['selection']['default_discard']['unclear_discarded'], r['selection']['default_discard']['discarded'],
                     number(r['calibration']['macro_f1'], 4)] for r in rows]),
             f"Selection macro F1 increases by {result['selection_f1_delta']:.4f}; teacher review-worthy misses "
             'fall from 132 to 67. Default discards fall by 44, adding review workload, while discarded unclear '
             'references increase from five to seventeen. The joint pattern is more informative than F1 alone. '
             'These development signals justify a fresh evaluation; they do not establish a new test-set winner.',
             '### Training composition and measured resources',
             table(['Training docs', 'Routine', 'Review-worthy', 'Unclear', 'Raw characters'],
                   [[r['training_documents'], *[r['training_classes'].get(k, 0) for k in LABELS], f"{r['training_characters']:,}"] for r in rows]),
             table(['Family', '4k documents', '16k documents'],
                   [[key, *[r['training_families'].get(key, 0) for r in rows]] for key in sorted(rows[1]['training_families'])]),
             'The 4k subset is an ordered prefix of the 16k subset, but source/class proportions change as '
             'smaller strata exhaust. Three epochs also mean 1,500 versus 6,000 optimizer steps. More documents, '
             'more updates and different composition change together; two points cannot establish a minimum '
             'label requirement or isolate the cause of the gain. Both fits additionally use 1,200 '
             'selection/calibration labels beyond their stated training size.',
             table(['Training docs', 'Train/validation min', 'Train peak GiB', 'Warm mean ms', 'Warm P95 ms', 'Inference peak GiB'],
                   [[r['training_documents'], number(r['training_seconds'] / 60, 1), number(r['training_peak_gib'], 2),
                     number(r['warm_mean_ms'], 2), number(r['warm_p95_ms'], 2), number(r['inference_peak_gib'], 2)] for r in rows]),
             f"The recorded train/validation clock grows {result['training_time_ratio']:.2f} times; warm mean latency "
             'is approximately unchanged on the identical 128-document timing sample. Same-recipe inference cost '
             'does not scale with the number of training examples. Loading and final calibration/resource work '
             f"are excluded from the training clock; the extension controller records {result['extension_wall_seconds'] / 3600:.2f} "
             'hours overall. The controller finished normally and released its temporary awake guard before '
             'the deadline. No cloud jobs, paid calls or additional fits ran.',
             '### Calibrated policies on development cohorts',
             'Thresholds are fitted on the 363-document calibration cohort. The last three columns show '
             'selection results, including discarded unclear references. Targets concern review-worthy '
             'misses, rather than all documents requiring review.',
             table(['Training docs', 'Target', 'Cal. discards / 363', 'Cal. misses / 252', 'Selection discards / 837', 'Selection misses / 552', 'Unclear discarded'],
                   [[r['training_documents'], key, r['calibration']['policies'][key]['discarded'],
                     r['calibration']['policies'][key]['dangerous_misses'], r['selection']['policies'][key]['discarded'],
                     r['selection']['policies'][key]['dangerous_misses'], r['selection']['policies'][key]['unclear_discarded']]
                    for r in rows for key in ('0.01', '0.05')]),
             'The 16k 1% policy misses eight of 552 teacher review-worthy selection references (1.45%), '
             'exceeding its empirical calibration target. Its 5% policy misses 21 (3.80%), while also '
             'discarding four unclear references. Target satisfaction on a fitting cohort is not a population '
             'guarantee. Policies were fitted on calibration alone; no rule was chosen using the original test outcomes.',
             'Independent verification checks ordered cohorts and nested training identities, source/runtime '
             'fingerprints, finite zero/best/final checkpoints, matched initialization, raw and calibrated '
             'arithmetic, calibration temperature and maximal feasible policies, selected epoch, timing sample '
             'and resource clocks. The completed 16k checkpoint was replayed on development cohorts only. '
             'The original study/result/ownership hashes remain unchanged. Published development probabilities, '
             'training metadata, decisions, histories and verifier receipts support CPU arithmetic replay '
             'without corpus bodies or model weights.']
    return parts


def compact_sections(result):
    models = result['models']
    names = {'bge_lora': 'BGE-small LoRA', 'finbert_lora': 'FinBERT LoRA', 'bge_frozen': 'Frozen BGE/logistic'}
    order = ('bge_lora', 'finbert_lora', 'bge_frozen')
    return ['', '## Development-only compact encoder comparison', '',
            'The user authorized one additional BGE-small LoRA fit after inspecting the original test results '
            'and the FinBERT development extension. It uses the same 4,000 training documents, 837 selection '
            'documents and 363 calibration documents as the 4k FinBERT recipe. All scores below use fitting '
            'cohorts; no new test prediction or 65th held-out comparison is included. The original study '
            'and its conclusions remain frozen. Fresh held-out data is required for generalization evidence.',
            'BGE uses pinned BAAI/bge-small-en-v1.5 weights, context 512, uniform whole-document pooling, '
            'rank 8, alpha 16, dropout 0.05, unweighted CE, both learning rates 0.0001, seed 42 and three '
            'epochs. Raw selection F1 chooses epoch two, retaining epoch zero as eligible. FinBERT selects '
            'epoch three under the same recipe. Architecture, pretrained objectives and trainable parameter '
            'counts differ; the frozen-feature classifier additionally has a different normalization and '
            'classifier recipe. These are system comparisons, not isolated tests of finance pretraining or LoRA.',
            table(['Method', 'Selection F1', 'Critical misses / 552', 'Unclear discarded / 93', 'Default discards / 837', 'Calibration F1'],
                  [[names[n], number(models[n]['selection']['macro_f1'], 4), models[n]['selection']['default_discard']['dangerous_misses'],
                    models[n]['selection']['default_discard']['unclear_discarded'], models[n]['selection']['default_discard']['discarded'],
                    number(models[n]['calibration']['macro_f1'], 4)] for n in order]),
            f"BGE LoRA's selection F1 is {result['f1_delta_vs_finbert']:.4f} relative to FinBERT and "
            f"{result['f1_delta_vs_frozen']:+.4f} relative to frozen BGE. Against FinBERT, it misses 48 fewer "
            'teacher review-worthy documents but sends 67 more documents for review and discards two '
            'additional unclear references. Against frozen BGE, it misses 25 fewer review-worthy documents '
            'while sending 23 more for review. The default operating points therefore trade workload '
            'for misses; the F1 differences alone do not establish operational superiority.',
            f"The matched BGE epoch-zero checkpoint has raw selection F1 {result['epoch_zero_selection']['macro_f1']:.4f}. "
            'Its improvement after training includes both the fresh classifier head and encoder adapters. '
            'An untrained triage head is a weak starting control; frozen BGE/logistic is the more useful '
            'trained alternative. Neither comparison isolates the causal benefit of adapter updates alone.',
            '### Local resource comparison',
            'Warm inference uses the identical 128-document selection sample and includes processing all '
            'source windows, excluding loading. Training clocks include validation but exclude startup '
            'and final calibration/resource measurement. Frozen BGE feature preparation and classifier '
            'fit use separate accounting, so no combined frozen-training clock is placed alongside these neural fits.',
            table(['Method', 'Train/validation min', 'Trainable parameters', 'Training peak GiB', 'Warm mean ms', 'Inference peak GiB'],
                  [[names[n], number(models[n]['training_seconds'] / 60, 1), f"{models[n]['trainable_parameters']:,}",
                    number(models[n]['training_peak_gib'], 2), number(models[n]['warm_mean_ms'], 2), number(models[n]['inference_peak_gib'], 2)]
                   for n in ('bge_lora', 'finbert_lora')]),
            f"BGE's observed warm mean is {result['warm_mean_ratio_vs_finbert']:.2f} times FinBERT's; "
            f"the bounded controller completes in {result['extension_wall_seconds'] / 60:.1f} minutes. Frozen "
            f"BGE/logistic's measured warm mean is {models['bge_frozen']['warm_mean_ms']:.2f} ms. "
            'These local measurements support a smaller resource footprint, not a guaranteed deployment '
            'speed or a finance-domain accuracy advantage. The controller finished normally and released '
            'its awake guard; no cloud jobs, paid calls or further training ran.',
            '### Development discard policies',
            'Thresholds fit calibration alone. The following table shows selection behavior; all three '
            'systems exceed both empirical calibration miss targets on this cohort. Targets concern '
            'review-worthy misses and do not protect all unclear documents.',
            table(['Method', 'Target', 'Selection discards / 837', 'Critical misses / 552', 'Miss rate', 'Unclear discarded'],
                  [[names[n], key, models[n]['selection']['policies'][key]['discarded'],
                    models[n]['selection']['policies'][key]['dangerous_misses'],
                    f"{100 * models[n]['selection']['policies'][key]['dangerous_miss_rate']:.2f}%",
                    models[n]['selection']['policies'][key]['unclear_discarded']] for n in order for key in ('0.01', '0.05')]),
            'Independent verification checks the exact shared 4k subset, original ordered cohorts, '
            'artifact/source/runtime identities, finite zero/best/final checkpoints, epoch-two selection, '
            'raw/calibrated arithmetic, calibration optimum and maximal feasible policies, and resource '
            'clocks/sample. Best and matched epoch-zero checkpoint probabilities were replayed on '
            'development data only. Published inputs retain original body ordering alongside a common '
            'identity order for comparison. All original core and completed FinBERT evidence hashes remain '
            'unchanged. These fitting-cohort findings motivate fresh evaluation; they carry no significance '
            'or independent generalization claim.']


def build(splits, study, human=None):
    plan = json.loads((study / "plan.json").read_text(encoding="utf-8"))
    freeze = validate_freeze(splits, study)
    results = json.loads((study / "results.json").read_text(encoding="utf-8"))
    for key, name in (("bootstrap", "bootstrap.json"), ("matched_coverage", "matched-coverage.json")):
        if fingerprint(study / name) != results[f"{key}_sha256"]:
            raise ValueError("Report statistics changed after test summarization")
    entries = freeze["entries"]
    predictions, configs, scored = {}, {}, {}
    for name, entry in entries.items():
        path = study / "test" / f"{name}.json"
        if fingerprint(path) != results["test_artifacts"][name]:
            raise ValueError("Test evidence changed before report generation")
        test = json.loads(path.read_text(encoding="utf-8"))
        if test["freeze_sha256"] != fingerprint(study / "freeze.json"):
            raise ValueError("Report test results use another freeze")
        predictions[name], scored[name] = test["predictions"], test["metrics"]["overall"]
        configs[name] = json.loads((study / entry["directory"] / "config.json").read_text(encoding="utf-8"))
    records = load_split(splits, "test")
    audit = human_audit(human, records, predictions, freeze["primary_linear"])
    if plan.get("preset") == "core" and human is None:
        audit["status"] = "Deferred for this version by agreement"
    quote = reference_quote(records)
    composition = training_composition(splits)
    bands = {name: uncertainty_bands(rows) for name, rows in predictions.items()}
    output = study / "report"
    output.mkdir(exist_ok=True)
    save(audit, output / "human-audit.json")
    save(quote, output / "reference-quote.json")
    save(composition, output / "training-composition.json")
    save({"definition": "1 minus maximum calibrated class probability",
          "method": "Fixed descriptive bands; argmax review-to-routine misses divided by review-worthy support within each band",
          "models": bands}, output / "uncertainty-bands.json")
    statistics = json.loads((study / "bootstrap.json").read_text())
    intervals = statistics["models"]
    main = {}
    for name, entry in entries.items():
        if entry["role"] in ("curve", "reference"):
            if plan.get("preset") == "core" and entry["role"] == "curve" and entry["samples"] != freeze["primary_samples"]:
                continue
            key = label(configs[name])
            if key not in main or entry["samples"] > entries[main[key]]["samples"]:
                main[key] = name
    primary, reference = freeze["primary_decision"], freeze["primary_linear"]
    names = list(dict.fromkeys((primary, reference, *main.values())))
    raw_scores = {name: json.loads((study / "test" / f"{name}.json").read_text())["raw_metrics"]["overall"] for name in names}
    has_qwen = any(config.get("kind", "").startswith("qwen") for config in configs.values())
    recipe_note = ("The bounded study adds one predeclared 4000-document LoRA recipe each for Laya, ModernBERT and FinBERT, "
                   "without a new hyperparameter sweep. Imported head recipes were previously selected at 1000/4000; "
                   "their tuning labels are additional to the stated training size."
                   if plan.get("preset") == "core" else
                   "Recipes are screened at 1000/4000 or nearest attainable sizes. Their tuning labels are additional, "
                   "so the smallest point is not a claim that its training rows alone suffice to discover the recipe.")
    figures(output, entries, configs, scored, intervals, names, (primary, reference))
    git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    delta = intervals[primary]["paired_delta_vs_reference"]["macro_f1"]
    conclusion = ("Adapted Laya improves teacher agreement over the selected linear baseline on this test cohort."
                  if delta["lower"] > 0 else "Adapted Laya agrees less well with the teacher than the selected linear baseline."
                  if delta["upper"] < 0 else "This test does not resolve a macro-F1 advantage for adapted Laya over the selected linear baseline.")
    parts = ["# Financial-document triage across public sources", "", conclusion,
             f"The primary validation-selected contrast is {label(configs[primary])} versus {label(configs[reference])}. "
             f"Test macro F1 is {scored[primary]['macro_f1']:.3f} versus {scored[reference]['macro_f1']:.3f}; "
             f"paired delta 95% interval {ci(delta)}. These are agreement scores against Luna labels.",
             f"Independent audit: {audit['status']}; {audit['resolved']} resolved and {audit['unresolved']} unresolved documents. "
             "No claim of analyst usefulness or investment profitability follows from teacher agreement alone.",
             "", "## Task and evaluation", "",
             "Classify supplied source content as routine, review_worthy or unclear for an analyst inbox. "
             "No issuer history, portfolio context, future returns or market reaction is supplied. Unclear routes to review. "
             "Macro F1 averages all three declared labels, including absent classes in small slices. "
             "The critical error is a review-worthy document predicted routine.",
             "The research question is whether supervised decision-model adaptation and larger contiguous context improve "
             "triage agreement, critical misses and throughput relative to lexical and encoder baselines. "
             "FinBERT's published sentiment results do not establish triage quality; BGE is an embedding model. "
             "ModernBERT supplies a backbone-family system comparison; classifier/decision heads and source-window budgets differ. "
             f"Primary sources and distinctions are documented in [literature.md](https://github.com/LuccaPereiraMartins/systematic-trading/blob/{git_revision}/decision_models/literature.md).",
             "Train ends 2026-03-31; selection/calibration use April-June; test uses July-September. "
             "Event and verified near-duplicate groups cannot cross boundaries. Old experimental bodies and near duplicates "
             "are excluded from fresh validation/test. Candidate retrieval is approximate and can miss duplicates. "
             "Every core recipe, checkpoint, calibration and discard threshold was frozen before test inference.",
             "", "## Source and label coverage", ""]
    if plan.get("preset") == "core":
        parts.insert(parts.index("## Task and evaluation") + 1,
                     "The primary contrast and main trained-model tables use 4000 training documents per method. "
                     "Unadapted references use no task training; larger imported curves are supplementary. "
                     "Generative LLM adaptation, larger LoRA runs and independent human review are deferred by the user. "
                     "This is an end-to-end teacher-agreement study, not expert validation of materiality.")
        primary_perf = json.loads((study / entries[primary]["directory"] / "performance.json").read_text())
        reference_perf = json.loads((study / entries[reference]["directory"] / "performance.json").read_text())
        primary_mean, reference_mean = [perf["latency_seconds"]["mean"] for perf in (primary_perf, reference_perf)]
        policy = scored[primary]["policies"]
        findings = ["## Findings and engineering implications",
                    f"The primary systems have {scored[primary]['default_discard']['dangerous_misses']} and "
                    f"{scored[reference]['default_discard']['dangerous_misses']} default critical misses, respectively, "
                    f"among {scored[primary]['default_discard']['review_worthy_support']} teacher review-worthy documents. "
                    f"Their warm mean latencies are {primary_mean * 1000:.1f} and {reference_mean * 1000:.2f} ms "
                    f"on the common selection sample ({primary_mean / reference_mean:.1f} times the latency for the decision model). "
                    "The small observed F1 difference and its interval make a strong lexical baseline essential to the comparison."]
        increment = statistics["contrasts"].get(primary + "-sample-increment")
        if increment:
            left, right = increment["left"], increment["right"]
            findings.append(f"Increasing Laya LoRA from {entries[right]['samples']} to {entries[left]['samples']} training "
                            f"documents raises observed F1 by {scored[left]['macro_f1'] - scored[right]['macro_f1']:.3f}; "
                            f"paired 95% interval {ci(increment['metrics']['macro_f1'])}. This supports sample efficiency "
                            "within this fixed recipe and cohort. Changing subset composition and additional selection/calibration "
                            "labels prevent interpreting it as a universal minimum-data requirement.")
        fin_lora = next((n for n in names if configs[n].get("kind") == "finbert" and configs[n].get("adaptation") == "lora"), None)
        fin_frozen = next((n for n in names if configs[n].get("kind") == "finbert" and configs[n]["family"] == "research_linear"), None)
        if fin_lora and fin_frozen:
            findings.append(f"FinBERT LoRA has observed F1 {scored[fin_lora]['macro_f1']:.3f} and "
                            f"{scored[fin_lora]['default_discard']['dangerous_misses']} critical misses, compared with "
                            f"{scored[fin_frozen]['macro_f1']:.3f} and {scored[fin_frozen]['default_discard']['dangerous_misses']} "
                            "for frozen FinBERT/logistic regression. These are point estimates for the declared recipes, "
                            "not an exhaustive fine-tuning comparison. Gains over a fresh random triage head alone do not "
                            "establish that adapters outperform a trained frozen-feature classifier.")
        findings += ["The completed head-context ablation does not show that wider context improves every metric. "
                     "F1 and critical misses need to be read together under the fixed pooling rule. These supplementary "
                     "comparisons do not justify selecting a new window size on the test cohort.",
                     f"The primary model's calibration-selected 1%/5% discard rules remove "
                     f"{100 * policy['0.01']['workload_reduction']:.2f}%/{100 * policy['0.05']['workload_reduction']:.2f}% "
                     f"of the test workload while missing {100 * policy['0.01']['dangerous_miss_rate']:.2f}%/"
                     f"{100 * policy['0.05']['dangerous_miss_rate']:.2f}% of teacher review-worthy documents. "
                     "Calibration targets are empirical constraints, not deployment bounds. The observed rates and support "
                     "matter more than the target names; confidence alone cannot certify safe automatic discarding.",
                     "This bounded study establishes an inspectable end-to-end comparison. Analyst usefulness and commercial-inbox "
                     "generalization still require independent labels and representative source coverage. Any further recipe or "
                     "policy selection needs fresh held-out data; the current test is closed to tuning."]
        index = parts.index("## Task and evaluation")
        parts[index:index] = findings
    backbones = {config["model"]: config["revision"] for config in configs.values() if config.get("model")}
    parts.insert(parts.index("## Source and label coverage") - 1,
                 table(["Pretrained model", "Pinned Hub revision"], sorted(backbones.items())))
    if "synthetic" in freeze["manifest"]["prepared"].get("method", "").lower():
        parts.insert(1, "Synthetic engineering fixture: this document verifies the pipeline and contains no research findings.")
    manifest = freeze["manifest"]["splits"]
    parts += corpus_tables(freeze["manifest"]["prepared"])
    families = sorted(set.union(*[set(v.get("families", {})) for v in manifest.values()]))
    parts += [table(["Family", "Train", "Selection", "Calibration", "Test"],
                    [[f, *[manifest[n].get("families", {}).get(f, 0) for n in ("train", "selection", "calibration", "test")]] for f in families]),
              "Source shortfalls are not filled by additional 8-Ks. These institutional/public-news cohorts are not a representative "
              "commercial inbox. Publication dates and snapshots cannot establish the exact text available at the historical date; "
              "pretraining exposure remains possible. Rights and collection methods are documented in "
              f"[sources.md](https://github.com/LuccaPereiraMartins/systematic-trading/blob/{git_revision}/decision_models/sources.md).",
              table(["Split", "Documents", *LABELS], [[n, manifest[n]["count"], *[manifest[n].get("labels", {}).get(k, 0) for k in LABELS]]
                                                     for n in ("train", "selection", "calibration", "test")]),
              "", "## Model comparisons", "",
              table(["Model", "Train", "Macro F1", "95% CI", "Misses", "Miss rate", "Brier"],
                    [[label(configs[n]), entries[n]["samples"], number(scored[n]["macro_f1"]), ci(intervals[n]["macro_f1"]),
                      scored[n]["default_discard"]["dangerous_misses"], number(scored[n]["default_discard"]["dangerous_miss_rate"]),
                      number(scored[n]["brier"])] for n in names]),
              "Misses use the default argmax routine decision; the denominator is review-worthy support. "
              "Brier sums squared errors over three classes (0-2). Full artifacts include class precision/recall/F1, confusion matrices, "
              "accuracy, log loss, ten-bin ECE, source slices and reference provenance. Intervals resample whole related groups "
              "2000 times with shared draws; they condition on the single trained seed and have no multiplicity correction.",
              "![Confusion matrices show the critical review-to-routine cell](confusion.png)",
              table(["Model", "Class", "Precision", "Recall", "F1", "Support"],
                    [[label(configs[n]), name, number(value['precision']), number(value['recall']), number(value['f1']), value['support']]
                     for n in (primary, reference) for name, value in scored[n]['per_class'].items()]),
              "", "## Source slices and uncertainty", "",
              table(["Model", "Family", "Documents", "Macro F1", "Review support", "Misses"],
                    [[label(configs[n]), family, value['count'], number(value['macro_f1']),
                      value['default_discard']['review_worthy_support'], value['default_discard']['dangerous_misses']]
                     for n in (primary, reference) for family, value in breakdown(predictions[n])['by_family'].items()]),
              table(["Model", "Provider", "Documents", "Macro F1", "Review support", "Misses"],
                    [[label(configs[n]), source, value['count'], number(value['macro_f1']),
                      value['default_discard']['review_worthy_support'], value['default_discard']['dangerous_misses']]
                     for n in (primary, reference) for source, value in breakdown(predictions[n])['by_source'].items()]),
              table(["Model", "Log loss", "ECE (10 bins)", "Median uncertainty"],
                    [[label(configs[n]), number(scored[n]['log_loss']), number(scored[n]['ece_10_bins']),
                      number(float(np.median([r['uncertainty'] for r in predictions[n]])))] for n in names]),
              table(["Model", "Raw Brier", "Cal. Brier", "Raw ECE", "Cal. ECE"],
                    [[label(configs[n]), number(raw_scores[n]['brier']), number(scored[n]['brier']),
                      number(raw_scores[n]['ece_10_bins']), number(scored[n]['ece_10_bins'])] for n in names]),
              "Raw and calibrated scores use the same test documents. A scalar temperature is fitted on separate "
              "calibration groups; it changes confidence and discard behavior while preserving argmax labels. "
              "Its use follows [Guo et al.](https://arxiv.org/abs/1706.04599); it does not guarantee calibration on this cohort.",
              "Uncertainty is 1 minus the largest calibrated class probability. It is model confidence, not a probability "
              "of investment materiality; teacher self-reported uncertainty is retained separately. Class support must be read "
              "beside small-slice macro F1. When no misses are observed, bootstrap resampling cannot reveal unseen errors.",
              *uncertainty_tables(bands, configs, (primary, reference)),
              "Bands are fixed before testing and descriptive, not selected discard thresholds. Boundary values enter the "
              "higher band; the final band includes its upper bound. Miss rates divide default review-to-routine misses by "
              "review-worthy support within each band. Empty bands and bands without review support have undefined rates, "
              "shown as dashes. Three-class uncertainty cannot exceed 2/3; the final band's upper bound covers all valid values.",
              "", "## Discard policies", ""]
    source_counts = {}
    for split in ("train", "selection", "calibration", "test"):
        counts = defaultdict(int)
        for row in load_split(splits, split):
            counts[row.get("source", "sec")] += 1
        source_counts[split] = counts
    providers = sorted(set.union(*[set(counts) for counts in source_counts.values()]))
    parts.insert(parts.index("## Model comparisons") - 1,
                 table(["Provider", "Train", "Selection", "Calibration", "Test"],
                       [[source, *[source_counts[split][source] for split in source_counts]] for source in providers]) +
                 "\n\nThe news family can contain official financial announcements as well as Wikinews. "
                 "Provider coverage is reported separately; official announcements do not establish commercial-journalism generalization.")
    coverage = freeze["manifest"].get("label_coverage", {})
    if coverage:
        parts.insert(parts.index("## Model comparisons") - 1,
                     table(["Label queue", "Queued", "Labeled", "Unlabeled", "Retained", "8-K cap excluded"],
                           [[name, value['queued'], value['labeled'], value['unlabeled'],
                             value.get('retained', value['labeled']), value.get('excluded_8k_share', 0)]
                            for name, value in coverage.items()]) +
                     "\n\nValidation/test labeling must be complete. A budget-limited training queue can retain unlabeled rows; "
                     "family/error counts remain in the manifest. The 40% primary 8-K cap is checked again on retained labels; "
                     "cap exclusions remain in the input queue and project spending. Attained sample sizes, rather than collection targets, define the learning curves.")
        if all("teacher_cost_usd" in value for value in coverage.values()):
            parts.insert(parts.index("## Model comparisons") - 1,
                         table(["Label queue", "Successful teacher labels", "Recorded USD", "Cost unavailable"],
                               [[name, value['successful_teacher_labels'], number(value['teacher_cost_usd'], 4),
                                 value['teacher_cost_unavailable']] for name, value in coverage.items()]) +
                         "\n\nThese costs cover successful teacher annotations retained in the frozen benchmark, "
                         "using recorded API usage and prices. Missing costs are counted separately. The shared $5 project ledger "
                         "also covers pilots, failed or unresolved requests and labels excluded from the benchmark; "
                         "these retained-label totals are not the entire project bill. Human-review time is excluded.")
    policy_rows = []
    for n in names:
        for key, value in scored[n]["policies"].items():
            policy_rows.append([label(configs[n]), key, number(value["workload_reduction"]), value["dangerous_misses"],
                                value["review_worthy_support"], number(value["dangerous_miss_rate"]), number(value["discard_contamination"])])
    parts += [table(["Model", "Target", "Workload reduction", "Misses", "Support", "Miss rate", "Contamination"], policy_rows),
              "Thresholds maximize discarded routine predictions subject to empirical calibration miss targets of 1%/5%. "
              "Predicted review-worthy/unclear always escalate. Sparse support and zero observed misses do not establish "
              "population guarantees. Test rates, rather than calibration targets, describe the observed deployment tradeoff.",
              "", "## Adaptation, sample efficiency and context", "",
              "![Nested learning curves on the same held-out cohort](learning-curves.png)",
              "Training sizes are nested 250/1000/4000/16000/all where attainable. Each neural run restarts from pinned pretrained "
              "weights. Deterministic family/label round-robin sampling changes class/source proportions as strata exhaust; "
              "the allocation must be considered when interpreting sample efficiency. One training seed is used. " + recipe_note,
              *training_tables(composition),
              "Counts use the exact frozen training subsets shared by all models. Independent groups may contain multiple "
              "related documents. These tables describe training allocation; recipe-screening and validation labels are additional.",
              "![Context ablations retain the same whole documents](context.png)",
              "Laya uses fixed uniform whole-document pooling and 50% overlapping windows; the native SDK reference uses "
              "most-confident-window aggregation. CE/Brier objectives are document-level; no chunk labels are invented. "
              "Matched pre-update controls separate supervised adaptation from aggregation and k-bit precision changes. "
              "FinBERT/BGE/ModernBERT use fresh triage heads; FinBERT's sentiment head is not mapped to triage labels.",
              "Laya's adapted wrapper averages native three-choice window logits. Literal reserved mask tokens are replaced "
              "with spaces before formatting. FinBERT/ModernBERT use non-overlapping source windows, average unpadded token "
              "states within each window, then uniformly average window vectors before a new LayerNorm/dropout/linear classifier. "
              "Frozen FinBERT/BGE baselines normalize the mean of all window vectors for logistic regression; BGE uses CLS "
              "rather than mean token states. Lexical baselines process full bodies. These systems differ in tokenization, "
              "window overlap and heads, so their differences do not isolate encoder architecture alone. LoRA updates Wqkv "
              "in Laya/ModernBERT and query/value projections in FinBERT: rank 8, alpha 16, adapter dropout 0.05."]
    if has_qwen:
        parts.append("Qwen uses non-thinking NF4/BF16 A/B/C scoring; QLoRA is conditional discriminative CE, not full-vocabulary SFT.")
    contrasts = statistics["contrasts"]
    parts += [table(["Adapted model", "Train", "Matched-zero F1", "Adapted F1", "Paired gain 95% CI"],
                    [[label(configs[n]), entries[n]['samples'], number(scored[entries[n]['matched_zero']]['macro_f1']),
                      number(scored[n]['macro_f1']), ci(contrasts[n + '-adaptation']['metrics']['macro_f1'])]
                     for n in names if entries[n].get('matched_zero')]),
              table(["Model", "Training increment", "F1 gain", "Paired gain 95% CI"],
                    [[label(configs[v['left']]), f"{entries[v['right']]['samples']} to {entries[v['left']]['samples']}",
                      number(scored[v['left']]['macro_f1'] - scored[v['right']]['macro_f1']), ci(v['metrics']['macro_f1'])]
                     for k, v in contrasts.items() if k.endswith('-sample-increment')]),
              table(["Model", "Context tokens", "F1 gain", "Paired gain 95% CI"],
                    [[label(configs[v['left']]), f"{configs[v['right']]['max_len']} to {configs[v['left']]['max_len']}",
                      number(scored[v['left']]['macro_f1'] - scored[v['right']]['macro_f1']), ci(v['metrics']['macro_f1'])]
                     for k, v in contrasts.items() if k.endswith('-context-increment')]),
              "Wider Laya windows change contiguous context and the number of votes under the same pooling rule. "
              "All encoded source tokens after tokenizer/reserved-token normalization remain covered at each context size; "
              "the difference is not previously omitted text.",
              "These paired differences describe this fixed-seed cohort. Additional tuning labels and multiple comparisons "
              "limit claims about minimum data requirements or statistical significance."]
    if has_qwen:
        matched = json.loads((study / "matched-coverage.json").read_text())
        parts += ["", "## Truncation and matched coverage", "",
                  f"Qwen's 4096-token budget includes the prompt. Long bodies retain first/last tokens, omitting the middle. "
                  f"The common fully covered subset contains {matched['documents']} documents, identical for every model.",
                  table(["Model", "Matched F1", "Review-worthy support", "Dangerous misses"],
                        [[label(configs[n]), number(matched['models'][n]['overall'].get('macro_f1')),
                          matched['models'][n]['overall'].get('default_discard', {}).get('review_worthy_support', 0),
                          matched['models'][n]['overall'].get('default_discard', {}).get('dangerous_misses', 0)] for n in names]),
                  "This subset can differ systematically from long documents; full-cohort results remain necessary. "
                  "Encoder/Laya windows retain all source tokens. Qwen probabilities are conditional on A/B/C; "
                  "full-vocabulary answer mass and retained-token fractions are saved per document."]
    parts += ["", "## Latency and cost", "", "![Warm throughput measured locally](throughput.png)"]
    timing = []
    for n in names:
        perf = json.loads((study / entries[n]["directory"] / "performance.json").read_text())
        timing.append([label(configs[n]), number(perf['latency_seconds']['p50']), number(perf['latency_seconds']['p95']),
                       number(perf['batch_throughput']['1']['documents_per_second']), number(perf['batch_throughput']['8']['documents_per_second']),
                       number(perf.get('peak_vram_bytes', 0) / 2**30, 2)])
    parts += [table(["Model", "P50 s", "P95 s", "Batch 1 docs/s", "Batch 8 docs/s", "Peak VRAM GiB"], timing),
              "Warm timings include preprocessing and all model windows, exclude loading, and use a fixed 128-document "
              "selection sample on a shared machine. Sklearn batches are native; neural/SDK/prompt document batches are serial, "
              "with window batching where supported. Local API charges are zero; hardware, electricity and collection are additional."]
    if plan.get("preset") != "core":
        parts += [
              "The paid GPT-6.1 Sol reference was not executed. Using checked Flex prices without cache discounts, this test cohort "
              "would cost the following input/output scenarios. o200k tokenization is an approximation; the separate byte bound "
              "reserves prompt overhead and is deliberately conservative.",
              table(["Output tokens/doc", "Approximate USD", "Conservative USD"],
                    [[k, number(v['approximate_usd'], 2), number(v['conservative_usd'], 2)] for k, v in quote['scenarios_output_tokens_per_document'].items()]),
              f"Pricing checked {quote['pricing_checked']}: [official pricing]({quote['pricing_source']}). Reasoning tokens count as output. "
              "A future paid run needs an updated concrete quote and authorization; it has no measured accuracy or latency here."]
    parts += ["", "## Critical-error inspection", ""]
    resources = []
    for n in names:
        directory = study / entries[n]["directory"]
        perf = json.loads((directory / "performance.json").read_text())
        history = json.loads((directory / "history.json").read_text()) if (directory / "history.json").exists() else {}
        weights = cached_weights(configs[n])
        resources.append([label(configs[n]), number(history.get('seconds', configs[n].get('fit_seconds', 0)), 1),
                          number(configs[n]['trainable_parameters'] / 1e6, 3) if 'trainable_parameters' in configs[n] else '-',
                          number(perf.get('training_peak_vram_bytes', 0) / 2**30, 2),
                          number(weights / 2**30 if weights is not None else None, 2),
                          number(perf.get('adapter_bytes', perf.get('artifact_bytes', 0)) / 2**20, 2)])
    parts.insert(parts.index("## Critical-error inspection") - 1,
                 table(["Model", "Fit elapsed s", "Trainable neural M", "Training VRAM GiB", "Original weights GiB", "Saved checkpoint MiB"], resources) +
                 "\n\nFit elapsed includes training/grid selection and recorded validation work; cached features can reduce elapsed time. "
                 "Neural trainable counts include heads and adapters. Equal LoRA rank does not imply equal updated parameter budgets: "
                 "Laya also trains its larger native decision head. Counts for linear classifiers are not recorded in this column. "
                 "Original weights are cached pretrained input files, not packed deployment size. "
                 "Saved checkpoints may include optimizer/RNG state. Peak VRAM is torch-allocated memory, excluding driver/display overhead; "
                 "the latency table reports inference peak, with training peak listed separately.")
    features = [(name, configs[name]["feature_preparation"]) for name in names if configs[name].get("feature_preparation")]
    if features:
        parts.insert(parts.index("## Critical-error inspection") - 1,
                     table(["Model", "Feature documents", "Recorded compute s", "Feature peak GiB", "Cache reused"],
                           [[label(configs[name]), sum(v['documents'] for v in values.values()),
                             number(sum(v['recorded_compute_seconds'] for v in values.values()), 1),
                             number(max(v['peak_vram_bytes'] for v in values.values()) / 2**30, 2),
                             sum(v['reused'] for v in values.values())] for name, values in features]) +
                     "\n\nFrozen-feature counts include train, selection and calibration. Recorded compute sums each requested "
                     "document's original preprocessing/encoder time under the pinned recipe; it excludes model loading and "
                     "cache I/O. Cached rows retain this measurement when reused across sample sizes. Fit elapsed includes "
                     "only preparation performed during that fit. These timings describe the shared machine, not a new cold run.")
    by_hash = {body_hash(r): r for r in records}
    misses = sorted([r for r in predictions[primary] if r['reference'] == 'review_worthy' and r['prediction'] == 'routine'],
                    key=lambda r: r['probabilities'][0], reverse=True)[:8]
    for r in misses:
        source = by_hash[r['body_sha256']]
        words = re.sub(r"\s+", " ", source['body']).split()
        excerpt = " ".join(words[:25]) + (" ..." if len(words) > 25 else "")
        parts += [f"Routine probability {r['probabilities'][0]:.3f}; {r['date']}; {r['family']}; "
                  f"[source]({source.get('url', '')}). Teacher label: review_worthy. Body hash: {r['body_sha256']}.", excerpt]
    if not misses:
        parts.append("No default review-worthy-to-routine misses were observed for the primary decision model on this finite cohort.")
    parts += ["Error examples are selected after testing for inspection, not model tuning. Labels may be wrong; "
              "analyst usefulness needs independent review. All miss identities and model disagreements remain inspectable in results.json.",
              "", "## Reproducibility and limits", "",
              "The serial runner stores failed/OOM configurations, any window-batch fallback, selected recipes, pinned model revisions, "
              "library versions, source snapshots, RNG/optimizer checkpoints, dataset/subset hashes and an immutable pre-test freeze. "
              "Reproduce with the locked dependencies and study.py --stage all; existing runs resume with changed-artifact guards. "
              "Raw-source rights/attribution travel with the corpus. Old API/container behavior remains separate.",
              "This is source-only inbox triage on constrained public cohorts, not expert materiality assessment or a trading strategy. "
              "One seed cannot quantify training instability. Bootstrap intervals quantify cohort sampling only. Teacher labels "
              "favor agreement with the labeling approach; independent audit availability is stated above. "
              "Missing source/date coverage, approximate duplicate retrieval and pretraining exposure restrict generalization. "
              "No paid SOTA comparison or RL/RLCD result is claimed. RL/RLCD remains a possible follow-on after the core study."]
    if audit["resolved"]:
        parts += ["", "## Independent non-expert audit", "",
                  f"Resolved {audit['resolved']} documents; unresolved {audit['unresolved']}; teacher agreement {number(audit['teacher_agreement'])}.",
                  table(["Model", "Human macro F1", "Dangerous misses", "Review-worthy support"],
                        [[label(configs[n]), number(audit['models'][n]['overall']['macro_f1']),
                          audit['models'][n]['overall']['default_discard']['dangerous_misses'],
                          audit['models'][n]['overall']['default_discard']['review_worthy_support']] for n in names])]
    diagnostic_outputs = ()
    inputs_path = output / "diagnostic-inputs.json"
    if inputs_path.exists():
        from diagnostics import analyze, figures as diagnostic_figures
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        if inputs['freeze_sha256'] != fingerprint(study / "freeze.json"):
            raise ValueError("Exploratory inputs belong to another frozen study")
        if (inputs['primary'], inputs['reference']) != (primary, reference):
            raise ValueError("Exploratory inputs cannot change the frozen primary comparison")
        selection_path = study / entries[reference]['directory'] / "selection.json"
        selection_rows = json.loads(selection_path.read_text(encoding="utf-8"))['predictions']
        if fingerprint(selection_path) != inputs['selection_sha256'] or [r['characters'] for r in selection_rows] != inputs['selection_characters']:
            raise ValueError("Exploratory length bands must use the unchanged selection cohort")
        diagnostic = analyze(predictions, inputs, statistics)
        save(diagnostic, output / "diagnostics.json")
        (output / "diagnostics-source.py").write_bytes((HERE / "diagnostics.py").read_bytes())
        labels = {n: label(configs[n]) for n in inputs['policy_models']}
        diagnostic_figures(output, diagnostic, labels)
        parts += exploratory_sections(diagnostic, labels, statistics)
        diagnostic_outputs = ('diagnostic-inputs.json', 'diagnostics.json', 'diagnostics-source.py',
                              'diagnostic-slices.png', 'diagnostic-policies.png')
    development_outputs = ()
    evidence_path = output / 'development-evidence.json.gz'
    if evidence_path.exists():
        import gzip
        from development import summarize
        evidence = json.loads(gzip.decompress(evidence_path.read_bytes()))
        if evidence['freeze_sha256'] != fingerprint(study / 'freeze.json'):
            raise ValueError('Development evidence belongs to another original study')
        development = summarize(evidence)
        save(development, output / 'development-summary.json')
        (output / 'development-source.py').write_bytes((HERE / 'development.py').read_bytes())
        parts += development_sections(development)
        development_outputs = ('development-evidence.json.gz', 'development-summary.json', 'development-source.py')
    compact_outputs = ()
    evidence_path = output / 'compact-evidence.json.gz'
    if evidence_path.exists():
        import gzip
        from compact import summarize
        evidence = json.loads(gzip.decompress(evidence_path.read_bytes()))
        if evidence['freeze_sha256'] != fingerprint(study / 'freeze.json'):
            raise ValueError('Compact development evidence belongs to another original study')
        compact = summarize(evidence)
        save(compact, output / 'compact-summary.json')
        (output / 'compact-source.py').write_bytes((HERE / 'compact.py').read_bytes())
        parts += compact_sections(compact)
        compact_outputs = ('compact-evidence.json.gz', 'compact-summary.json', 'compact-source.py')
    content = "\n\n".join(parts) + "\n"
    (output / "report.md").write_text(content, encoding="utf-8")
    (output / "report-source.py").write_bytes((HERE / "report.py").read_bytes())
    pdf(content, output / "report.pdf")
    save({"freeze_sha256": fingerprint(study / "freeze.json"), "results_sha256": fingerprint(study / "results.json"),
          "report_source_sha256": fingerprint(HERE / "report.py"),
          "versions": {n: version(n) for n in ('matplotlib', 'reportlab', 'pypdf')},
          "outputs": {name: fingerprint(output / name) for name in
                      ('report.md', 'report.pdf', 'report-source.py', 'human-audit.json', 'reference-quote.json', 'training-composition.json', 'uncertainty-bands.json',
                       'learning-curves.png', 'context.png', 'throughput.png', 'confusion.png') + diagnostic_outputs + development_outputs + compact_outputs}}, output / "manifest.json")
    print(f"Report written to {output}; render and inspect PDF before publication", flush=True)
