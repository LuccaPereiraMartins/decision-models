"""Extend the original cohort without moving old evaluation groups; freeze binary references."""

import argparse
from collections import Counter, defaultdict
from datetime import date
from itertools import zip_longest
import hashlib
import json
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from prepare import CUTOFFS, NEAR_THRESHOLD, PERMUTATIONS, SEARCH_THRESHOLD, cluster, family, partition, sample, shingle_set
from schemas import BINARY_LABELS, BINARY_TASK, HERE, annotation, body_hash, file_hash, load_split, read_records, save, write_records


NAMES = ("train", "selection", "calibration", "test")
COUNTS = {"train": 25000, "selection": 1000, "calibration": 500, "test": 2000}


def rows_from(path):
    if path.suffix == ".json":
        yield from read_records(path)
    else:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def group_order(groups):
    """Round-robin family/month; give the additional ECB coverage first priority."""
    buckets = defaultdict(list)
    for key, rows in groups.items():
        buckets[(min(family(r) for r in rows), min(r["date"][:7] for r in rows))].append(key)
    result = []
    for ecb_first in (True, False):
        keys = sorted(k for k in buckets if (k[0] == "ecb") == ecb_first)
        ordered = [sorted(buckets[bucket]) for bucket in keys]
        result.extend(key for round_ in zip_longest(*ordered) for key in round_ if key is not None)
    return result


def whole_groups(groups, count):
    """Exact subset-sum with stable first-solution precedence; never split a group."""
    if count < 0:
        raise ValueError("The requested count cannot remove existing evaluation documents")
    if count == 0:
        return set()
    reachable = {0: ()}
    for key in group_order(groups):
        size = len(groups[key])
        for total, chosen in sorted(list(reachable.items()), reverse=True):
            target = total + size
            if target <= count and target not in reachable:
                reachable[target] = (*chosen, key)
        if count in reachable:
            return set(reachable[count])
    raise ValueError(f"Cannot obtain exactly {count} documents from {sum(map(len, groups.values()))} whole-group candidates")


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    return dict(groups)


def signature(body):
    result = MinHash(num_perm=PERMUTATIONS, seed=42)
    shingles = sorted(shingle_set(body))
    for start in range(0, len(shingles), 4096):
        result.update_batch(shingles[start:start + 4096])
    return result


def exclude_near(candidates, base):
    """LSH retrieves candidates; exact five-token Jaccard verifies every exclusion."""
    by_hash = {body_hash(row): row for row in base}
    index = MinHashLSH(threshold=SEARCH_THRESHOLD, num_perm=PERMUTATIONS)
    for number, (key, row) in enumerate(by_hash.items(), 1):
        index.insert(key, signature(row["body"]))
        if number % 1000 == 0:
            print(f"Binary duplicate index: {number}/{len(by_hash)}", flush=True)
    accepted, matches = [], []
    for number, row in enumerate(candidates, 1):
        shingles = shingle_set(row["body"])
        hits = []
        for key in index.query(signature(row["body"])):
            other = shingle_set(by_hash[key]["body"])
            similarity = len(shingles & other) / len(shingles | other)
            if similarity >= NEAR_THRESHOLD:
                hits.append((key, similarity))
        if hits:
            key, similarity = sorted(hits)[0]
            matches.append({"body_sha256": body_hash(row), "base_body_sha256": key, "jaccard": similarity})
        else:
            accepted.append(row)
        if number % 250 == 0:
            print(f"Binary candidate near audit: {number}/{len(candidates)}", flush=True)
    return accepted, matches


def binary_input(row, original_partition=None):
    old = annotation(row)["label"] if original_partition else None
    result = {**row, "previous_llm_annotation": row["llm"], "previous_human_annotation": row["human"],
              "reference_protocol": {"task_id": BINARY_TASK, "original_label": old,
                  "basis": "retained_three_class" if old in BINARY_LABELS else ("binary_redecision" if original_partition else "new_binary_label")}}
    if original_partition:
        result["original_partition"] = original_partition
    if old not in BINARY_LABELS:
        result.update(llm={"label": None, "uncertainty": None}, human={"label": None, "uncertainty": None})
    result.pop("label_error", None)
    return result


def verify_inputs(prepared, output):
    for path, expected in prepared["inputs"].items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f"Prepared source changed: {path}")
    for name, expected in prepared["prepared_input_sha256"].items():
        if file_hash(output / f"{name}-input.jsonl") != expected:
            raise ValueError(f"Prepared cohort changed: {name}")


def validate_cohorts(groups, original, counts):
    body_sets, group_sets = {}, {}
    for name, rows in groups.items():
        if len(rows) != counts[name]:
            raise ValueError(f"Expected exactly {counts[name]} {name} documents")
        body_sets[name], group_sets[name] = {body_hash(r) for r in rows}, {r["group_id"] for r in rows}
        if len(body_sets[name]) != len(rows):
            raise ValueError(f"Duplicate body within {name}")
        expected = "validation" if name in ("selection", "calibration") else name
        if any(partition(r["date"]) != expected for r in rows):
            raise ValueError(f"A publication date crosses the {name} boundary")
        if name != "train":
            mapping = {body_hash(r): r["group_id"] for r in rows}
            if any(mapping.get(body_hash(r)) != r["group_id"] for r in original[name]):
                raise ValueError(f"Original {name} membership or group identity changed")
    for i, left in enumerate(NAMES):
        for right in NAMES[i + 1:]:
            if body_sets[left] & body_sets[right] or group_sets[left] & group_sets[right]:
                raise ValueError(f"Body or event group crosses {left}/{right}")


def prepare(original, corpora, ecb, ecb_audit, output, legacy, legacy_events, counts=None):
    counts = counts or COUNTS
    corpora = sorted(corpora)
    settings = {"original": str(original.resolve()), "corpora": [str(p.resolve()) for p in corpora],
                "ecb": str(ecb.resolve()), "ecb_audit": str(ecb_audit.resolve()),
                "legacy": str(legacy.resolve()), "legacy_events": str(legacy_events.resolve()), "counts": counts}
    if (output / "manifest.json").exists():
        raise ValueError("Binary benchmark is frozen; use another output directory")
    if (output / "prepared.json").exists():
        prepared = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
        verify_inputs(prepared, output)
        if prepared["settings"] != settings:
            raise ValueError("Preparation settings changed")
        print("Existing preparation verified; reuse its labeling queue", flush=True)
        return prepared
    audit = json.loads(ecb_audit.read_text(encoding="utf-8"))
    expected = audit.get("corpus_sha256", audit.get("files", {}).get(ecb.name))
    if audit.get("status") not in ("pass", "passed", "verified") or expected != file_hash(ecb):
        raise ValueError("Need a finalized successful ECB audit binding its corpus SHA")
    old_manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    original_rows = {name: load_split(original, name) for name in NAMES}
    legacy_rows = read_records(legacy)
    registry = json.loads(legacy_events.read_text(encoding="utf-8"))
    if registry["legacy_sha256"] != file_hash(legacy) or registry.get("status") != "complete":
        raise ValueError("Legacy registry is incomplete or references another corpus")
    inputs = [original / "manifest.json", original / "subsets.json", legacy, legacy_events, ecb, ecb_audit,
              HERE / "prepare.py", Path(__file__).resolve(), *corpora,
              *[original / f"{name}.jsonl" for name in NAMES]]
    hashes = {str(path.resolve()): file_hash(path) for path in inputs}
    for path in [*corpora, legacy, legacy_events]:
        if hashes[str(path.resolve())] != old_manifest["prepared"]["inputs"].get(str(path.resolve())):
            raise ValueError(f"Original raw corpus changed: {path}")
    base = [*legacy_rows, *[row for rows in original_rows.values() for row in rows]]
    blocked_bodies = {body_hash(row) for row in base}
    blocked_bodies.update(m["body_sha256"] for m in old_manifest["prepared"]["legacy_extraction_check"]["matches"])
    blocked_events = {row["event_id"] for row in base if row.get("event_id")} | set(registry["events"])
    candidates, seen, exclusions, event_dates = [], set(), Counter(), defaultdict(set)
    for path in [*corpora, ecb]:
        for row in rows_from(path):
            if date.fromisoformat(row["date"]).isoformat() != row["date"]:
                raise ValueError("Need ISO publication dates")
            period, key, event = partition(row["date"]), body_hash(row), row.get("event_id")
            if event:
                event_dates[event].add(period)
            if period == "outside" or (period == "train" and path != ecb):
                continue
            if key in blocked_bodies or event in blocked_events:
                exclusions["existing_body_or_event"] += 1
            elif key in seen:
                exclusions["candidate_exact_duplicate"] += 1
            else:
                seen.add(key)
                candidates.append(row)
    cross_dates = {event for event, periods in event_dates.items() if len(periods) > 1}
    exclusions["raw_event_cross_boundary"] = sum(r.get("event_id") in cross_dates for r in candidates)
    candidates = [r for r in candidates if r.get("event_id") not in cross_dates]
    candidates.sort(key=lambda r: (r["date"], body_hash(r)))
    candidates, near_matches = exclude_near(candidates, base)
    exclusions["verified_near_existing"] = len(near_matches)
    identities = cluster(candidates, set())
    additions = defaultdict(list)
    for i, row in enumerate(candidates):
        additions[identities[i][0]].append({**row, "group_id": identities[i][0], "previously_experimented": False})
    available = defaultdict(dict)
    for key, rows in additions.items():
        periods = {partition(r["date"]) for r in rows}
        if len(periods) != 1:
            exclusions["new_group_cross_boundary"] += len(rows)
        else:
            available[periods.pop()][key] = rows
    train_add = [row for rows in available["train"].values() for row in rows]
    if any(r.get("source") != "ecb" for r in train_add):
        raise ValueError("Only additional ECB documents may replace old training rows")
    removable = {key: rows for key, rows in grouped(original_rows["train"]).items()
                 if all(r.get("source", "sec") == "sec" for r in rows)}
    removed = whole_groups(removable, len(train_add))
    chosen = {"train": [r for r in original_rows["train"] if r["group_id"] not in removed] + train_add}
    development = dict(available["validation"])
    for name in ("selection", "calibration", "test"):
        pool = available["test"] if name == "test" else development
        selected = whole_groups(pool, counts[name] - len(original_rows[name]))
        chosen[name] = original_rows[name] + [r for key in sorted(selected) for r in pool[key]]
        for key in selected:
            pool.pop(key)
    validate_cohorts(chosen, original_rows, counts)
    output.mkdir(parents=True, exist_ok=True)
    details = {}
    for name, rows in chosen.items():
        old = {body_hash(r) for r in original_rows[name]}
        converted = [binary_input(r, name if body_hash(r) in old else None) for r in sorted(rows, key=lambda r: (r["date"], body_hash(r)))]
        write_records(converted, output / f"{name}-input.jsonl")
        details[name] = {"count": len(rows), "new_documents": sum(body_hash(r) not in old for r in rows),
                         "sources": dict(Counter(r.get("source", "sec") for r in rows)),
                         "pending": sum(annotation(r)["label"] is None for r in converted)}
    prepared = {"task_id": BINARY_TASK, "labels": list(BINARY_LABELS), "counts": counts,
                "original_directory": str(original.resolve()), "settings": settings, "inputs": hashes, "cutoffs": list(CUTOFFS),
                "prepared_input_sha256": {name: file_hash(output / f"{name}-input.jsonl") for name in NAMES},
                "partitions": details, "exclusions": dict(exclusions), "near_matches": near_matches,
                "removed_train_groups": sorted(removed), "ecb_audit_sha256": file_hash(ecb_audit),
                "original_eval_memberships_sha256": {name: digest([(body_hash(r), r["group_id"]) for r in original_rows[name]]) for name in NAMES[1:]},
                "near_duplicate_threshold": NEAR_THRESHOLD, "candidate_threshold": SEARCH_THRESHOLD,
                "method": "Inherited evaluation groups plus label-blind whole-group temporal additions; binary references",
                "audit_limits": "LSH candidate retrieval is probabilistic. New candidates checked against selected original cohorts and legacy bodies; original group audit inherited. Unselected raw corpora are not fully re-clustered."}
    save(prepared, output / "prepared.json")
    print(json.dumps(details), flush=True)
    return prepared


def reference_metadata(row):
    return {key: value for key, value in row.items() if key not in ("llm", "human", "label_error")}


def freeze(output, label_files, retained_audit=None):
    from label import BINARY_RUBRIC_HASH
    if (output / "manifest.json").exists():
        raise ValueError("Binary benchmark already frozen; never replace its references")
    prepared = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
    verify_inputs(prepared, output)
    original = Path(prepared["original_directory"])
    old_rows = {name: load_split(original, name) for name in NAMES}
    labels, evidence = {}, {}
    for path in label_files:
        rows = read_records(path)
        evidence[str(path.resolve())] = file_hash(path)
        for row in rows:
            value = annotation(row)
            if value["label"] is None:
                continue
            if value["label"] not in BINARY_LABELS or value.get("rubric_sha256") != BINARY_RUBRIC_HASH or value.get("task_id") != BINARY_TASK:
                raise ValueError("New references must use the frozen binary rubric and task")
            key = body_hash(row)
            if key in labels and labels[key] != row:
                raise ValueError("Conflicting binary labels or metadata for the same body")
            labels[key] = row
    groups = {}
    for name in NAMES:
        rows = list(rows_from(output / f"{name}-input.jsonl"))
        original_by_hash = {body_hash(r): r for r in old_rows[name]}
        for row in rows:
            key = body_hash(row)
            if key in original_by_hash and row != binary_input(original_by_hash[key], name):
                raise ValueError("Prepared original references or provenance disagree with the original frozen row")
            if key not in original_by_hash and row["reference_protocol"]["basis"] != "new_binary_label":
                raise ValueError("New documents cannot claim retained original references")
            if row["reference_protocol"]["basis"] != "retained_three_class":
                if body_hash(row) not in labels:
                    raise ValueError(f"Missing binary label in {name}: {body_hash(row)}")
                queued = labels[key]
                if reference_metadata(row) != reference_metadata(queued) or queued["human"].get("label") is not None:
                    raise ValueError("Binary label queue changed prepared body/source/date/group/provenance metadata")
                row["llm"] = queued["llm"]
            if annotation(row)["label"] not in BINARY_LABELS:
                raise ValueError("No unclear or unlabeled references may enter the binary benchmark")
        groups[name] = rows
    validate_cohorts(groups, old_rows, prepared["counts"])
    consistency = None
    if retained_audit:
        audit_rows = read_records(retained_audit)
        old_train = {body_hash(r): r for r in old_rows["train"]}
        matrix = {a: {b: 0 for b in BINARY_LABELS} for a in BINARY_LABELS}
        if len(audit_rows) != 200:
            raise ValueError("Need exactly 200 completed blinded consistency audit documents")
        if len({body_hash(r) for r in audit_rows}) != len(audit_rows):
            raise ValueError("Consistency audit contains duplicate bodies")
        for row in audit_rows:
            key, value = body_hash(row), row["llm"]
            original_row = old_train.get(key)
            original_label = annotation(original_row)["label"] if original_row else None
            if original_label not in BINARY_LABELS or value["label"] not in BINARY_LABELS or value.get("rubric_sha256") != BINARY_RUBRIC_HASH or value.get("task_id") != BINARY_TASK or row["human"].get("label") is not None:
                raise ValueError("Consistency audit needs completed binary re-labels of retained training references")
            if row.get("original_partition") != "train" or row.get("previous_llm_annotation") != original_row["llm"] or row.get("previous_human_annotation") != original_row["human"] or row.get("reference_protocol") != {"task_id": BINARY_TASK, "basis": "retained_label_consistency_audit", "original_label": original_label}:
                raise ValueError("Consistency audit changed original reference provenance")
            if any(row.get(k) != v for k, v in reference_metadata(original_row).items()):
                raise ValueError("Consistency audit changed original training document metadata")
            matrix[original_label][value["label"]] += 1
        consistency = {"documents": len(audit_rows), "confusion_matrix": matrix,
                       "agreement": sum(matrix[k][k] for k in BINARY_LABELS) / len(audit_rows),
                       "sha256": file_hash(retained_audit), "used_to_override_references": False}
    if retained_audit is None and prepared["counts"] == COUNTS:
        raise ValueError("The real binary study freeze requires the completed 200-document consistency audit")
    manifest = {"task_id": BINARY_TASK, "labels": list(BINARY_LABELS), "method": prepared["method"],
                "prepared": prepared, "label_sources_sha256": evidence, "retained_label_consistency": consistency,
                "test_reuse_warning": "Original three-class test outcomes were previously inspected; revised binary cohort is not a fresh independent confirmation.",
                "splits": {}}
    files = {}
    for name, rows in groups.items():
        content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        files[f"{name}.jsonl"] = content
        manifest["splits"][name] = {"count": len(rows), "start": min(r["date"] for r in rows), "end": max(r["date"] for r in rows),
                                    "sha256": hashlib.sha256(content).hexdigest(),
                                    "families": dict(Counter(family(r) for r in rows)),
                                    "sources": dict(Counter(r.get("source", "sec") for r in rows)),
                                    "labels": dict(Counter(annotation(r)["label"] for r in rows)),
                                    "reference_bases": dict(Counter(r["reference_protocol"]["basis"] for r in rows))}
    ordered = sample(groups["train"], by_label=True)
    sizes = sorted({n for n in (4000, 16000, len(ordered)) if n <= len(ordered)})
    subsets = {"task_id": BINARY_TASK, "labels": list(BINARY_LABELS), "seed": 42,
               "order": "Round-robin family/binary-label; stable body-hash ordering",
               "subsets": {str(n): [body_hash(r) for r in ordered[:n]] for n in sizes}}
    files["subsets.json"] = (json.dumps(subsets, indent=2) + "\n").encode()
    manifest["subsets_sha256"] = hashlib.sha256(files["subsets.json"]).hexdigest()
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    for name, content in files.items():
        if (output / name).exists() and (output / name).read_bytes() != content:
            raise ValueError(f"Refuse to replace a partially frozen artifact: {name}")
    for name, content in files.items():
        (output / name).write_bytes(content)
    print(json.dumps(manifest["splits"]), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=HERE / "data/research/benchmark")
    parser.add_argument("--output", type=Path, default=HERE / "data/research/binary-v2")
    parser.add_argument("--corpora", type=Path, nargs="+", default=list((HERE / "data/research/corpus").glob("*.jsonl")))
    parser.add_argument("--ecb", type=Path, default=HERE / "data/research/binary-v2-corpus/ecb.jsonl")
    parser.add_argument("--ecb-audit", type=Path, default=HERE / "data/research/binary-v2-corpus/ecb-verification.json")
    parser.add_argument("--legacy", type=Path, default=HERE / "data/dataset.json")
    parser.add_argument("--legacy-events", type=Path, default=HERE / "data/research/legacy-events/events.json")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--labels", type=Path, nargs="+", default=[])
    parser.add_argument("--retained-audit", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.output, args.labels, args.retained_audit)
    else:
        prepare(args.original, args.corpora, args.ecb, args.ecb_audit, args.output, args.legacy, args.legacy_events)
