#!/usr/bin/env python3
"""Paired PDB-cluster bootstrap for two micro-AAR per-record tables."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_lmpro(path, topk):
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t")
                if int(row["topk"]) == topk]
    return [
        {
            "source_line": int(row["source_line"]),
            "pdb": row["pdb"].lower(),
            "matched": int(row["succ"]),
            "total": int(row["tot"]),
        }
        for row in rows
    ]


def load_proteinmpnn(path, topk):
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t")
                if int(row["topk"]) == topk]
    return [
        {
            "source_line": int(row["source_line"]),
            "pdb": row["pdb"].lower(),
            "condition": row["cdr_backbone_condition"],
            "matched": int(row["matched_positions"]),
            "total": int(row["total_positions"]),
        }
        for row in rows
    ]


def _aggregate_by_unit(rows, resampling_unit):
    if resampling_unit not in {"pdb", "record"}:
        raise ValueError(f"unsupported resampling unit: {resampling_unit}")
    labels = [
        row["pdb"] if resampling_unit == "pdb" else str(row["source_line"])
        for row in rows
    ]
    ordered_labels = list(dict.fromkeys(labels))
    grouped = []
    for label in ordered_labels:
        indices = [index for index, value in enumerate(labels) if value == label]
        grouped.append(
            {
                "label": label,
                "matched": sum(rows[index]["matched"] for index in indices),
                "total": sum(rows[index]["total"] for index in indices),
            }
        )
    return grouped


def _summary(point, samples):
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "point_estimate": float(point),
        "bootstrap_95_ci": [float(lower), float(upper)],
    }


def bootstrap_aar(rows, resamples, seed, resampling_unit="pdb"):
    if not rows:
        raise ValueError("the table must contain at least one record")
    grouped = _aggregate_by_unit(rows, resampling_unit)
    matches = np.asarray([row["matched"] for row in grouped], dtype=np.int64)
    totals = np.asarray([row["total"] for row in grouped], dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        samples[index] = matches[sampled].sum() / totals[sampled].sum()
    return {
        "records": len(rows),
        "clusters": len(grouped),
        "resampling_unit": resampling_unit,
        "resamples": resamples,
        "seed": seed,
        "aar": _summary(matches.sum() / totals.sum(), samples),
    }


def bootstrap_comparison(
    lmpro, proteinmpnn, resamples, seed, resampling_unit="pdb"
):
    if len(lmpro) != len(proteinmpnn) or not lmpro:
        raise ValueError("the paired tables must contain the same nonzero record count")
    for index, (left, right) in enumerate(zip(lmpro, proteinmpnn)):
        if left["source_line"] != right["source_line"]:
            raise ValueError(
                f"source record differs at row {index}: "
                f"{left['source_line']} != {right['source_line']}"
            )
        if left["pdb"] != right["pdb"]:
            raise ValueError(
                f"record order differs at row {index}: {left['pdb']} != {right['pdb']}"
            )
        if left["total"] != right["total"]:
            raise ValueError(
                f"evaluated-position counts differ at row {index}: "
                f"{left['total']} != {right['total']}"
            )
    conditions = {row["condition"] for row in proteinmpnn}
    if len(conditions) != 1:
        raise ValueError(f"ProteinMPNN rows mix backbone conditions: {conditions}")

    lm_grouped = _aggregate_by_unit(lmpro, resampling_unit)
    protein_grouped = _aggregate_by_unit(proteinmpnn, resampling_unit)
    if [row["label"] for row in lm_grouped] != [row["label"] for row in protein_grouped]:
        raise ValueError("paired tables produce different resampling clusters")
    lm_matches = np.asarray([row["matched"] for row in lm_grouped], dtype=np.int64)
    protein_matches = np.asarray(
        [row["matched"] for row in protein_grouped], dtype=np.int64
    )
    totals = np.asarray([row["total"] for row in lm_grouped], dtype=np.int64)
    protein_totals = np.asarray(
        [row["total"] for row in protein_grouped], dtype=np.int64
    )
    if not np.array_equal(totals, protein_totals):
        raise ValueError("paired tables have different totals within resampling clusters")
    rng = np.random.default_rng(seed)
    lm_samples = np.empty(resamples, dtype=np.float64)
    protein_samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(lm_grouped), size=len(lm_grouped))
        denominator = totals[sampled].sum()
        lm_samples[index] = lm_matches[sampled].sum() / denominator
        protein_samples[index] = protein_matches[sampled].sum() / denominator
    deltas = lm_samples - protein_samples

    denominator = totals.sum()
    return {
        "records": len(lmpro),
        "clusters": len(lm_grouped),
        "resampling_unit": resampling_unit,
        "proteinmpnn_cdr_backbone_condition": next(iter(conditions)),
        "resamples": resamples,
        "seed": seed,
        "lmpro": _summary(lm_matches.sum() / denominator, lm_samples),
        "proteinmpnn": _summary(
            protein_matches.sum() / denominator, protein_samples
        ),
        "delta_lmpro_minus_proteinmpnn": {
            **_summary(
                (lm_matches.sum() - protein_matches.sum()) / denominator, deltas
            ),
            "bootstrap_probability_above_zero": float(np.mean(deltas > 0)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmpro-tsv", type=Path, required=True)
    parser.add_argument("--proteinmpnn-tsv", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument(
        "--resampling-unit", choices=("pdb", "record"), default="pdb",
        help="bootstrap PDB clusters by default; record is a sensitivity analysis",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.topk <= 0 or args.resamples <= 0:
        parser.error("--topk and --resamples must be positive")

    result = bootstrap_comparison(
        load_lmpro(args.lmpro_tsv, args.topk),
        load_proteinmpnn(args.proteinmpnn_tsv, args.topk),
        args.resamples,
        args.seed,
        args.resampling_unit,
    )
    result.update(
        {
            "topk": args.topk,
            "lmpro_tsv": str(args.lmpro_tsv),
            "proteinmpnn_tsv": str(args.proteinmpnn_tsv),
        }
    )
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)


if __name__ == "__main__":
    main()
