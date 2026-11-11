#!/usr/bin/env python3
"""Merge record-disjoint ProteinMPNN metric shards using count-weighted aggregates."""

import argparse
import csv
import json
from pathlib import Path


def merge_payloads(payloads, sources):
    conditions = {payload.get("cdr_backbone_condition") for payload in payloads}
    if len(conditions) != 1 or None in conditions:
        raise ValueError(f"metric shards disagree on backbone condition: {conditions}")
    topk_sets = [{row["topk"] for row in payload["results"]} for payload in payloads]
    if any(topks != topk_sets[0] for topks in topk_sets[1:]):
        raise ValueError("metric shards have different top-k sets")

    merged_results = []
    for topk in sorted(topk_sets[0]):
        rows = [
            next(row for row in payload["results"] if row["topk"] == topk)
            for payload in payloads
        ]
        candidate_counts = {row["candidates_per_record"] for row in rows}
        if len(candidate_counts) != 1:
            raise ValueError(f"top-{topk} shards disagree on candidate count")
        matched = sum(row["matched_positions"] for row in rows)
        total = sum(row["total_positions"] for row in rows)
        selected = sum(row["selected_designs"] for row in rows)
        records = sum(row["records"] for row in rows)
        score_sum = sum(
            row["mean_proteinmpnn_score"] * row["selected_designs"] for row in rows
        )
        merged_results.append(
            {
                "topk": topk,
                "records": records,
                "candidates_per_record": next(iter(candidate_counts)),
                "selected_designs": selected,
                "matched_positions": matched,
                "total_positions": total,
                "aar": matched / total,
                "mean_proteinmpnn_score": score_sum / selected,
            }
        )
    return {
        "cdr_backbone_condition": next(iter(conditions)),
        "shard_metrics": [str(source) for source in sources],
        "results": merged_results,
    }


def merge_per_record_rows(rows_by_shard, payloads, condition):
    expected_fields = {
        "source_line",
        "name",
        "pdb",
        "cdr_backbone_condition",
        "topk",
        "matched_positions",
        "total_positions",
        "aar",
        "mean_proteinmpnn_score",
    }
    merged = []
    observed_keys = set()
    for rows, payload in zip(rows_by_shard, payloads):
        for row in rows:
            if set(row) != expected_fields:
                raise ValueError("per-record table has unexpected columns")
            source_line = int(row["source_line"])
            topk = int(row["topk"])
            key = (source_line, topk)
            if key in observed_keys:
                raise ValueError(f"duplicate per-record key: {key}")
            observed_keys.add(key)
            if row["cdr_backbone_condition"] != condition:
                raise ValueError(f"per-record row {key} has the wrong backbone condition")
            merged.append(row)

        by_topk = {result["topk"]: result for result in payload["results"]}
        for topk, result in by_topk.items():
            selected = [row for row in rows if int(row["topk"]) == topk]
            if len(selected) != result["records"]:
                raise ValueError(f"top-{topk} per-record row count disagrees with metrics")
            matched = sum(int(row["matched_positions"]) for row in selected)
            total = sum(int(row["total_positions"]) for row in selected)
            if matched != result["matched_positions"] or total != result["total_positions"]:
                raise ValueError(f"top-{topk} per-record counts disagree with metrics")
    return sorted(merged, key=lambda row: (int(row["source_line"]), int(row["topk"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-record-inputs", type=Path, nargs="+")
    parser.add_argument("--per-record-output", type=Path)
    args = parser.parse_args()
    if (args.per_record_inputs is None) != (args.per_record_output is None):
        parser.error("--per-record-inputs and --per-record-output must be used together")
    if args.per_record_inputs is not None and len(args.per_record_inputs) != len(args.inputs):
        parser.error("per-record input count must match metric input count")
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    merged = merge_payloads(payloads, args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    for row in merged["results"]:
        print(
            f"top {row['topk']:>4}: AAR = {100.0 * row['aar']:.2f}% "
            f"({row['matched_positions']}/{row['total_positions']}; "
            f"{row['records']} records)"
        )

    if args.per_record_inputs is not None:
        rows_by_shard = []
        for path in args.per_record_inputs:
            with path.open(newline="") as handle:
                rows_by_shard.append(list(csv.DictReader(handle, delimiter="\t")))
        rows = merge_per_record_rows(
            rows_by_shard, payloads, merged["cdr_backbone_condition"]
        )
        args.per_record_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_record_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.per_record_output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
