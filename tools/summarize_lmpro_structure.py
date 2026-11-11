#!/usr/bin/env python3
"""Summarize explicitly named LmPro CDR-H3 structure metrics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "local_cdr3_ca_rmsd",
    "framework_anchored_cdr3_ca_rmsd",
    "framework_anchored_cdr3_backbone_rmsd",
)


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
    }


def summarize_rows(rows):
    if not rows:
        raise ValueError("structure TSV is empty")
    if len({row["source_line"] for row in rows}) != len(rows):
        raise ValueError("structure TSV contains duplicate source_line values")
    missing = [metric for metric in METRICS if metric not in rows[0]]
    if missing:
        raise ValueError(f"structure TSV is missing metrics: {missing}")
    return {
        "records": len(rows),
        "metric_definitions": {
            "local_cdr3_ca_rmsd": "CDR-H3 CA after CDR-local Kabsch alignment",
            "framework_anchored_cdr3_ca_rmsd":
                "CDR-H3 CA without alignment in the supplied native-framework frame",
            "framework_anchored_cdr3_backbone_rmsd":
                "CDR-H3 N/CA/C without alignment in the supplied native-framework frame",
        },
        "metrics": {
            metric: summarize_values([float(row[metric]) for row in rows])
            for metric in METRICS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = summarize_rows(rows)
    result["source_tsv"] = str(args.input)
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)


if __name__ == "__main__":
    main()
