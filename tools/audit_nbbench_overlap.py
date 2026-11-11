#!/usr/bin/env python
"""Audit exact NbBench VHH overlap with the INDI pretraining train corpus.

The INDI corpus is only available as a fairseq indexed dataset.  This tool groups
both corpora by sequence length and compares compact encoded byte strings, avoiding
materialising eleven million Python strings.  CDRInfilling is compared using its
unmasked target sequence (the ``label`` column), not the masked model input.
"""

import argparse
import collections
import csv
import json
import os
import time

import numpy as np
from fairseq.data import Dictionary, data_utils


DATASETS = {
    "VRClassification": "seq",
    "CDRInfilling": "label",
    "thermo-seq": "seq",
    "thermo-tm": "seq",
    "vhh_affinity-seq": "seq",
    "vhh_affinity-score": "seq",
    "polyreaction": "seq",
    "nanobody_type": "seq",
    "SARS-CoV-2": "VHH_sequence",
    "hIL6": "VHH_sequence",
    "Paratope": "seq_nanobody",
}
SPLITS = ("train", "val", "test")


def normalize_sequence(value):
    return "".join(value.split()).upper()


def read_nbbench(data_dir):
    rows = []
    unique_sequences = set()
    for dataset, sequence_column in DATASETS.items():
        for split in SPLITS:
            path = os.path.join(data_dir, dataset, f"{split}.csv")
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle)
                if sequence_column not in (reader.fieldnames or []):
                    raise ValueError(f"{path} has no {sequence_column!r} column")
                for row_index, row in enumerate(reader):
                    sequence = normalize_sequence(row[sequence_column])
                    if not sequence:
                        raise ValueError(f"empty sequence in {path}, data row {row_index}")
                    rows.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "row_index": row_index,
                            "sequence": sequence,
                        }
                    )
                    unique_sequences.add(sequence)
    return rows, unique_sequences


def encode_targets(sequences, dictionary):
    symbol_to_id = {dictionary[index]: index for index in range(len(dictionary))}
    targets_by_length = collections.defaultdict(dict)
    unsupported = {}
    for sequence in sorted(sequences):
        try:
            encoded = bytes(symbol_to_id[residue] for residue in sequence)
        except KeyError as error:
            unsupported[sequence] = error.args[0]
            continue
        targets_by_length[len(sequence)][encoded] = sequence
    return targets_by_length, unsupported


def scan_pretraining(pretrain_dir, targets_by_length, batch_size, max_records=None):
    dictionary = Dictionary.load(os.path.join(pretrain_dir, "seq", "dict.txt"))
    dataset = data_utils.load_indexed_dataset(
        os.path.join(pretrain_dir, "seq", "train"), dictionary, None
    )
    record_count = len(dataset) if max_records is None else min(len(dataset), max_records)
    sizes = dataset.sizes[:record_count].astype(np.int64)
    starts = dataset._index._pointers[:record_count] // dataset._index._dtype_size
    sequence_data = np.memmap(
        os.path.join(pretrain_dir, "seq", "train.bin"),
        mode="r",
        dtype=dataset._index._dtype,
    )

    counts = collections.Counter()
    started = time.time()
    for length in sorted(targets_by_length):
        # fairseq-preprocess appends one EOS token to each raw sequence.
        matching_starts = starts[sizes == length + 1]
        target_map = targets_by_length[length]
        target_keys = np.array(list(target_map), dtype=f"V{length}")
        matched_occurrences = 0
        positions = np.arange(length, dtype=np.int64)
        for begin in range(0, len(matching_starts), batch_size):
            offsets = matching_starts[begin : begin + batch_size]
            indices = offsets[:, None] + positions
            tokens = np.ascontiguousarray(sequence_data[indices], dtype=np.uint8)
            keys = tokens.view(f"V{length}").reshape(-1)
            hit = np.isin(keys, target_keys)
            if not hit.any():
                continue
            values, value_counts = np.unique(keys[hit], return_counts=True)
            for value, count in zip(values, value_counts):
                sequence = target_map[bytes(value)]
                counts[sequence] += int(count)
                matched_occurrences += int(count)
        print(
            f"length={length:3d} targets={len(target_map):6d} "
            f"corpus_records={len(matching_starts):8d} "
            f"matches={matched_occurrences:8d} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    return counts, dictionary, record_count, len(dataset)


def summarize(rows, counts, unsupported):
    summary = []
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["split"])].append(row)
    for dataset in DATASETS:
        for split in SPLITS:
            group = groups[(dataset, split)]
            unique = {row["sequence"] for row in group}
            overlapping_rows = sum(bool(counts[row["sequence"]]) for row in group)
            overlapping_unique = sum(bool(counts[sequence]) for sequence in unique)
            unsupported_rows = sum(row["sequence"] in unsupported for row in group)
            summary.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "rows": len(group),
                    "unique_sequences": len(unique),
                    "exact_overlap_rows": overlapping_rows,
                    "exact_overlap_row_fraction": overlapping_rows / len(group),
                    "exact_overlap_unique_sequences": overlapping_unique,
                    "exact_overlap_unique_fraction": overlapping_unique / len(unique),
                    "unsupported_rows": unsupported_rows,
                }
            )
    return summary


def write_outputs(output_dir, rows, counts, unsupported, summary, metadata):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "exact_overlap_rows.tsv")
    with open(manifest_path, "w", newline="") as handle:
        fieldnames = (
            "dataset",
            "split",
            "row_index",
            "sequence",
            "pretrain_occurrences",
            "unsupported_residue",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            sequence = row["sequence"]
            writer.writerow(
                {
                    **row,
                    "pretrain_occurrences": counts[sequence],
                    "unsupported_residue": unsupported.get(sequence, ""),
                }
            )

    positive_path = os.path.join(output_dir, "exact_overlap_positive_rows.tsv")
    with open(positive_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            sequence = row["sequence"]
            if not counts[sequence]:
                continue
            writer.writerow(
                {
                    **row,
                    "pretrain_occurrences": counts[sequence],
                    "unsupported_residue": unsupported.get(sequence, ""),
                }
            )

    summary_path = os.path.join(output_dir, "exact_overlap_summary.tsv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)

    report_path = os.path.join(output_dir, "exact_overlap_report.json")
    with open(report_path, "w") as handle:
        json.dump({"metadata": metadata, "summary": summary}, handle, indent=2)
    return manifest_path, positive_path, summary_path, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="baselines/nbbench/data")
    parser.add_argument("--pretrain-dir", default="abgnn/final_3_data")
    parser.add_argument("--output-dir", default="baselines/nbbench/overlap_audit")
    parser.add_argument("--batch-records", type=int, default=50_000)
    parser.add_argument(
        "--max-pretrain-records",
        type=int,
        default=None,
        help="smoke-test only; omit for the publishable full-corpus audit",
    )
    args = parser.parse_args()

    rows, sequences = read_nbbench(args.data_dir)
    dictionary = Dictionary.load(os.path.join(args.pretrain_dir, "seq", "dict.txt"))
    targets_by_length, unsupported = encode_targets(sequences, dictionary)
    print(
        f"NbBench rows={len(rows)}, unique VHHs={len(sequences)}, "
        f"encoded={sum(map(len, targets_by_length.values()))}, unsupported={len(unsupported)}",
        flush=True,
    )
    counts, _, scanned_records, corpus_records = scan_pretraining(
        args.pretrain_dir,
        targets_by_length,
        args.batch_records,
        args.max_pretrain_records,
    )
    summary = summarize(rows, counts, unsupported)
    metadata = {
        "data_dir": os.path.abspath(args.data_dir),
        "pretrain_dir": os.path.abspath(args.pretrain_dir),
        "pretrain_records_scanned": scanned_records,
        "pretrain_records_total": corpus_records,
        "complete_corpus_scan": scanned_records == corpus_records,
        "unique_nbbench_sequences": len(sequences),
        "unsupported_unique_sequences": len(unsupported),
    }
    outputs = write_outputs(
        args.output_dir, rows, counts, unsupported, summary, metadata
    )
    print("\nExact full-sequence overlap")
    for item in summary:
        print(
            f"{item['dataset']:20s} {item['split']:5s} "
            f"rows={item['exact_overlap_rows']:7d}/{item['rows']:<7d} "
            f"unique={item['exact_overlap_unique_sequences']:7d}/"
            f"{item['unique_sequences']:<7d}"
        )
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
