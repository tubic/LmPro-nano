#!/usr/bin/env python
"""Evaluate a leakage-audited nearest-context CDR3 homology baseline."""

import argparse
import collections
import csv
import json
import os

import numpy as np
from Bio.Align import PairwiseAligner


def load_subset(path):
    if path is None:
        return None
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                records.append(int(line.split()[0]))
    if not records:
        raise ValueError(f"empty record subset: {path}")
    if records != sorted(set(records)):
        raise ValueError(f"record subset is not unique and sorted: {path}")
    return records


def parse_entry(entry, source_line):
    sequence = entry["antibody_seq"]
    tags = entry["antibody_cdr"]
    if len(sequence) != len(tags):
        raise ValueError(f"line {source_line}: sequence/tag lengths differ")
    cdr3_indices = [index for index, tag in enumerate(tags) if tag == "3"]
    if not cdr3_indices:
        raise ValueError(f"line {source_line}: missing CDR3")
    if cdr3_indices != list(range(cdr3_indices[0], cdr3_indices[-1] + 1)):
        raise ValueError(f"line {source_line}: non-contiguous CDR3")
    start = cdr3_indices[0]
    end = cdr3_indices[-1] + 1
    return {
        "source_line": source_line,
        "pdb": entry["pdb"].lower(),
        "sequence": sequence,
        "cdr3": sequence[start:end],
        "cdr3_length": end - start,
        "context": sequence[:start] + sequence[end:],
    }


def load_entries(path, subset):
    subset_set = set(subset) if subset is not None else None
    entries = []
    with open(path) as handle:
        for source_line, line in enumerate(handle):
            if subset_set is not None and source_line not in subset_set:
                continue
            entries.append(parse_entry(json.loads(line), source_line))
    if subset is not None and [entry["source_line"] for entry in entries] != subset:
        found = {entry["source_line"] for entry in entries}
        raise ValueError(f"records missing from {path}: {sorted(set(subset) - found)}")
    return entries


def build_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -3.0
    aligner.extend_gap_score = -0.5
    return aligner


def alignment_identity(aligner, query, target):
    alignment = aligner.align(query, target)[0]
    matches = 0
    for (query_begin, query_end), (target_begin, target_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        length = min(query_end - query_begin, target_end - target_begin)
        for offset in range(length):
            matches += query[query_begin + offset] == target[target_begin + offset]
    return matches / max(len(query), len(target))


def positional_consensus(entries):
    by_length = collections.defaultdict(list)
    for entry in entries:
        by_length[entry["cdr3_length"]].append(entry["cdr3"])
    consensus = {}
    for length, sequences in by_length.items():
        consensus[length] = "".join(
            collections.Counter(sequence[position] for sequence in sequences)
            .most_common(1)[0][0]
            for position in range(length)
        )
    return consensus


def recovery(prediction, target):
    if len(prediction) != len(target):
        raise ValueError("recovery requires equal sequence lengths")
    return sum(left == right for left, right in zip(prediction, target))


def summarize(rows, prefix):
    matches = sum(row[f"{prefix}_matches"] for row in rows)
    positions = sum(row["cdr3_length"] for row in rows)
    exact = sum(row[f"{prefix}_prediction"] == row["native_cdr3"] for row in rows)
    macro = np.mean(
        [row[f"{prefix}_matches"] / row["cdr3_length"] for row in rows]
    )
    return {
        "micro_aar": matches / positions,
        "macro_aar": float(macro),
        "exact_cdr3_rate": exact / len(rows),
        "matches": matches,
        "positions": positions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-jsonl",
        default="abgnn/finetune/exp2-hern/nanobody_clustered/train_data.jsonl",
    )
    parser.add_argument(
        "--test-jsonl",
        default="abgnn/finetune/exp2-hern/nanobody_clustered/test_data.jsonl",
    )
    parser.add_argument(
        "--train-records",
        default="baselines/nanobody_clustered/proteinmpnn_clean/train_records_lmpro_loadable.txt",
    )
    parser.add_argument(
        "--test-records",
        default="baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt",
    )
    parser.add_argument(
        "--output-dir", default="baselines/nanobody_clustered/homology_clean"
    )
    args = parser.parse_args()

    train = load_entries(args.train_jsonl, load_subset(args.train_records))
    test = load_entries(args.test_jsonl, load_subset(args.test_records))
    train_by_length = collections.defaultdict(list)
    train_cdr3_counts = collections.Counter()
    for entry in train:
        train_by_length[entry["cdr3_length"]].append(entry)
        train_cdr3_counts[entry["cdr3"]] += 1
    consensus = positional_consensus(train)
    aligner = build_aligner()
    rows = []
    for test_index, target in enumerate(test, start=1):
        candidates = train_by_length[target["cdr3_length"]]
        if not candidates:
            raise ValueError(
                f"no train CDR3 with length {target['cdr3_length']} for test line "
                f"{target['source_line']}"
            )
        best = None
        best_score = -float("inf")
        for candidate in candidates:
            score = aligner.score(target["context"], candidate["context"])
            score /= max(len(target["context"]), len(candidate["context"]))
            if score > best_score:
                best = candidate
                best_score = score
        nearest_prediction = best["cdr3"]
        consensus_prediction = consensus[target["cdr3_length"]]
        rows.append(
            {
                "test_source_line": target["source_line"],
                "pdb": target["pdb"],
                "cdr3_length": target["cdr3_length"],
                "native_cdr3": target["cdr3"],
                "nearest_train_source_line": best["source_line"],
                "nearest_context_score": best_score,
                "nearest_context_identity": alignment_identity(
                    aligner, target["context"], best["context"]
                ),
                "nearest_prediction": nearest_prediction,
                "nearest_matches": recovery(nearest_prediction, target["cdr3"]),
                "consensus_prediction": consensus_prediction,
                "consensus_matches": recovery(consensus_prediction, target["cdr3"]),
                "train_exact_cdr3_occurrences": train_cdr3_counts[target["cdr3"]],
            }
        )
        if test_index == 1 or test_index % 25 == 0:
            print(f"evaluated {test_index}/{len(test)}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "per_record.tsv")
    with open(result_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "train_records": len(train),
        "test_records": len(test),
        "protocol": (
            "same-CDR3-length training template with maximum globally aligned "
            "non-CDR3 context score; ties retain the lowest source line"
        ),
        "nearest_context_homology": summarize(rows, "nearest"),
        "same_length_positional_consensus": summarize(rows, "consensus"),
        "test_cdr3_seen_exactly_in_train": sum(
            row["train_exact_cdr3_occurrences"] > 0 for row in rows
        ),
        "mean_nearest_context_identity": float(
            np.mean([row["nearest_context_identity"] for row in rows])
        ),
        "median_nearest_context_identity": float(
            np.median([row["nearest_context_identity"] for row in rows])
        ),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
