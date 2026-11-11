#!/usr/bin/env python
"""Audit exact downstream CDR3 overlap with the binarized pretraining corpus.

The clustered downstream split prevents train/validation/test leakage, but that alone
does not prove that a pretrained encoder has never seen a test target. The INDI corpus
ships only in fairseq's mmap format, so this tool compares its sequence and CDR-tag
datasets directly without requiring the omitted raw token files.

Example:

    python tools/audit_pretrain_overlap.py \
        --split-dir abgnn/finetune/exp2-hern/nanobody_clustered \
        --restrict-records baselines/nanobody_clustered/eval_subset_records.txt \
        --write-disjoint-records \
            baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt

The generated subset removes records whose exact native CDR3 occurs in the pretraining
*training* partition. It is an exact-match control, not a homology-cluster audit.
"""

import argparse
import collections
import json
import os
import time

import numpy as np
from fairseq.data import Dictionary, data_utils


def load_record_subset(path):
    if path is None:
        return None
    records = set()
    with open(path) as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                records.add(int(line.split()[0]))
    return records


def native_cdr3(record):
    return "".join(
        residue
        for residue, tag in zip(record["antibody_seq"], record["antibody_cdr"])
        if tag == "3"
    )


def locate_cdr3_spans(tag_data, cdr3_id, chunk_tokens):
    """Return flattened [start, end) offsets for every contiguous CDR3 span."""
    starts, ends = [], []
    total = len(tag_data)
    found = 0
    started = time.time()
    for chunk_index, begin in enumerate(range(0, total, chunk_tokens)):
        end = min(total, begin + chunk_tokens)
        tags = np.asarray(tag_data[begin:end])
        is_cdr3 = tags == cdr3_id
        previous_is_cdr3 = bool(tag_data[begin - 1] == cdr3_id) if begin else False
        next_is_cdr3 = bool(tag_data[end] == cdr3_id) if end < total else False
        starts_here = is_cdr3 & np.r_[not previous_is_cdr3, ~is_cdr3[:-1]]
        ends_here = is_cdr3 & np.r_[~is_cdr3[1:], not next_is_cdr3]
        start_offsets = np.flatnonzero(starts_here).astype(np.int64) + begin
        end_offsets = np.flatnonzero(ends_here).astype(np.int64) + begin + 1
        starts.append(start_offsets)
        ends.append(end_offsets)
        found += len(start_offsets)
        if chunk_index % 5 == 0 or end == total:
            print(
                f"scan tags {end / total:.1%}: spans={found}, "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    starts = np.concatenate(starts)
    ends = np.concatenate(ends)
    if len(starts) != len(ends):
        raise RuntimeError("CDR3 start/end counts differ in the pretraining tag corpus")
    return starts, ends


def count_exact_spans(sequence_data, starts, ends, targets, symbol_to_id, batch_size):
    """Count exact target strings among variable-length contiguous corpus spans."""
    targets_by_length = collections.defaultdict(dict)
    for target in targets:
        try:
            encoded = bytes(symbol_to_id[residue] for residue in target)
        except KeyError as exc:
            raise ValueError(f"target CDR3 contains unknown residue {exc.args[0]!r}") from exc
        targets_by_length[len(target)][encoded] = target

    lengths = ends - starts
    counts = collections.Counter()
    for length in sorted(targets_by_length):
        matching_starts = starts[lengths == length]
        target_map = targets_by_length[length]
        target_keys = np.array(list(target_map), dtype=f"V{length}")
        for begin in range(0, len(matching_starts), batch_size):
            offsets = matching_starts[begin : begin + batch_size]
            indices = offsets[:, None] + np.arange(length, dtype=np.int64)
            tokens = np.ascontiguousarray(sequence_data[indices], dtype=np.uint8)
            keys = tokens.view(f"V{length}").reshape(-1)
            hit = np.isin(keys, target_keys)
            if not hit.any():
                continue
            values, value_counts = np.unique(keys[hit], return_counts=True)
            for value, count in zip(values, value_counts):
                counts[target_map[bytes(value)]] += int(count)
        print(
            f"length={length:2d}: corpus_spans={len(matching_starts):8d}, "
            f"matched_occurrences={sum(counts[t] for t in target_map.values()):8d}",
            flush=True,
        )
    return counts


def count_exact_full_sequences(
    sequence_data, dataset, records, cdr3_counts, symbol_to_id, batch_size
):
    """Count exact full antibody matches for records with an overlapping CDR3."""
    targets_by_length = collections.defaultdict(dict)
    for record in records:
        if not cdr3_counts[native_cdr3(record)]:
            continue
        sequence = record["antibody_seq"]
        try:
            encoded = bytes(symbol_to_id[residue] for residue in sequence)
        except KeyError:
            continue
        targets_by_length[len(sequence)][encoded] = sequence

    starts = dataset._index._pointers // dataset._index._dtype_size
    lengths = dataset.sizes.astype(np.int64) - 1  # fairseq appends EOS
    counts = collections.Counter()
    for length in sorted(targets_by_length):
        matching_starts = starts[lengths == length]
        target_map = targets_by_length[length]
        target_keys = np.array(list(target_map), dtype=f"V{length}")
        for begin in range(0, len(matching_starts), batch_size):
            offsets = matching_starts[begin : begin + batch_size]
            indices = offsets[:, None] + np.arange(length, dtype=np.int64)
            tokens = np.ascontiguousarray(sequence_data[indices], dtype=np.uint8)
            keys = tokens.view(f"V{length}").reshape(-1)
            hit = np.isin(keys, target_keys)
            if not hit.any():
                continue
            values, value_counts = np.unique(keys[hit], return_counts=True)
            for value, count in zip(values, value_counts):
                counts[target_map[bytes(value)]] += int(count)
    return counts


def write_disjoint_subset(path, source_path, base_records, records, cdr3_counts):
    kept = [
        index
        for index in sorted(base_records)
        if cdr3_counts[native_cdr3(records[index])] == 0
    ]
    removed = len(base_records) - len(kept)
    with open(path, "w") as handle:
        handle.write("# Exact-pretraining-CDR3-disjoint evaluation subset\n")
        handle.write(f"# Derived from {source_path}\n")
        handle.write(
            f"# {len(kept)} records; removed {removed} records whose native CDR3 "
            "occurs in the INDI pretraining train partition.\n"
        )
        handle.write("# This controls exact overlap only; it is not a homology audit.\n")
        for index in kept:
            handle.write(f"{index}\n")
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", default="abgnn/final_3_data")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--restrict-records", default=None)
    parser.add_argument("--write-disjoint-records", default=None)
    parser.add_argument("--chunk-tokens", type=int, default=50_000_000)
    parser.add_argument("--batch-spans", type=int, default=200_000)
    parser.add_argument("--full-sequence-batch", type=int, default=20_000)
    args = parser.parse_args()

    jsonl_path = os.path.join(args.split_dir, f"{args.split}_data.jsonl")
    with open(jsonl_path) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    restricted = load_record_subset(args.restrict_records)
    base_records = restricted if restricted is not None else set(range(len(records)))
    if not base_records <= set(range(len(records))):
        raise SystemExit("--restrict-records contains an index outside the downstream split")

    targets = collections.defaultdict(list)
    for index, record in enumerate(records):
        targets[native_cdr3(record)].append(index)

    sequence_dictionary = Dictionary.load(
        os.path.join(args.pretrain_dir, "seq", "dict.txt")
    )
    tag_dictionary = Dictionary.load(os.path.join(args.pretrain_dir, "tag", "dict.txt"))
    sequence_dataset = data_utils.load_indexed_dataset(
        os.path.join(args.pretrain_dir, "seq", "train"), sequence_dictionary, None
    )
    tag_dataset = data_utils.load_indexed_dataset(
        os.path.join(args.pretrain_dir, "tag", "train"), tag_dictionary, None
    )
    if len(sequence_dataset) != len(tag_dataset) or not np.array_equal(
        sequence_dataset.sizes, tag_dataset.sizes
    ):
        raise SystemExit("pretraining sequence and tag datasets are not aligned")

    sequence_data = np.memmap(
        os.path.join(args.pretrain_dir, "seq", "train.bin"),
        mode="r",
        dtype=sequence_dataset._index._dtype,
    )
    tag_data = np.memmap(
        os.path.join(args.pretrain_dir, "tag", "train.bin"),
        mode="r",
        dtype=tag_dataset._index._dtype,
    )
    symbol_to_id = {
        sequence_dictionary[index]: index for index in range(len(sequence_dictionary))
    }

    print(
        f"pretrain records={len(sequence_dataset)}, downstream records={len(records)}, "
        f"restricted records={len(base_records)}, unique downstream CDR3s={len(targets)}"
    )
    starts, ends = locate_cdr3_spans(
        tag_data, tag_dictionary.index("3"), args.chunk_tokens
    )
    cdr3_counts = count_exact_spans(
        sequence_data,
        starts,
        ends,
        targets,
        symbol_to_id,
        args.batch_spans,
    )
    full_counts = count_exact_full_sequences(
        sequence_data,
        sequence_dataset,
        records,
        cdr3_counts,
        symbol_to_id,
        args.full_sequence_batch,
    )

    overlapping_targets = {target for target in targets if cdr3_counts[target]}
    overlapping_all_records = [
        index for index, record in enumerate(records) if cdr3_counts[native_cdr3(record)]
    ]
    overlapping_restricted = [
        index
        for index in sorted(base_records)
        if cdr3_counts[native_cdr3(records[index])]
    ]
    full_overlap_records = [
        index
        for index, record in enumerate(records)
        if full_counts[record["antibody_seq"]]
    ]

    print("\nExact-overlap summary")
    print(
        f"  unique CDR3s: {len(overlapping_targets)}/{len(targets)} "
        f"({len(overlapping_targets) / len(targets):.2%})"
    )
    print(
        f"  all downstream records: {len(overlapping_all_records)}/{len(records)} "
        f"({len(overlapping_all_records) / len(records):.2%})"
    )
    print(
        f"  restricted records: {len(overlapping_restricted)}/{len(base_records)} "
        f"({len(overlapping_restricted) / len(base_records):.2%})"
    )
    print(f"  exact full-antibody records: {len(full_overlap_records)}/{len(records)}")
    print("\nOverlapping records")
    for index in overlapping_all_records:
        record = records[index]
        cdr3 = native_cdr3(record)
        marker = "yes" if index in base_records else "no"
        print(
            f"  source_index={index} pdb={record['pdb']} restricted={marker} "
            f"cdr3={cdr3} pretrain_occurrences={cdr3_counts[cdr3]} "
            f"full_occurrences={full_counts[record['antibody_seq']]}"
        )

    if args.write_disjoint_records:
        source = args.restrict_records or f"all source indices in {jsonl_path}"
        kept = write_disjoint_subset(
            args.write_disjoint_records,
            source,
            base_records,
            records,
            cdr3_counts,
        )
        print(
            f"\nwrote {len(kept)} exact-disjoint records to "
            f"{args.write_disjoint_records}"
        )


if __name__ == "__main__":
    main()
