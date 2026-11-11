#!/usr/bin/env python3
"""Add stable source-line identity to legacy LmPro per-record TSV output."""

import argparse
import csv
import json
from pathlib import Path


def load_selected(path):
    selected = []
    for raw in path.read_text().splitlines():
        value = raw.split("#", 1)[0].strip()
        if value:
            selected.append(int(value.split()[0]))
    return selected


def load_source_pdbs(path):
    with path.open() as handle:
        return [json.loads(line)["pdb"].lower() for line in handle if line.strip()]


def backfill_rows(rows, selected, source_pdbs):
    if not rows:
        raise ValueError("per-record TSV is empty")
    topks = sorted({int(row["topk"]) for row in rows})
    if len(rows) != len(selected) * len(topks):
        raise ValueError(
            f"expected {len(selected) * len(topks)} rows for "
            f"{len(selected)} records x {len(topks)} top-k values, got {len(rows)}"
        )

    output = []
    for record_index, source_line in enumerate(selected):
        begin = record_index * len(topks)
        group = rows[begin:begin + len(topks)]
        expected_pdb = source_pdbs[source_line]
        observed_pdbs = {row["pdb"].lower() for row in group}
        observed_topks = {int(row["topk"]) for row in group}
        if observed_pdbs != {expected_pdb} or observed_topks != set(topks):
            raise ValueError(
                f"row group {record_index} does not match source line {source_line}: "
                f"pdb={observed_pdbs}, topk={observed_topks}"
            )
        for row in group:
            converted = {"source_line": str(source_line), **row}
            if "rmsd" in converted:
                converted["local_cdr3_ca_rmsd"] = converted.pop("rmsd")
            output.append(converted)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--restrict-records", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    converted = backfill_rows(
        rows,
        load_selected(args.restrict_records),
        load_source_pdbs(args.source_jsonl),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=converted[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(converted)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
