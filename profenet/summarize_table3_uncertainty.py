#!/usr/bin/env python3
"""Report fold variation and complex-cluster bootstrap intervals for Table 3."""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from rde.utils.skempi import (
    overall_auroc,
    overall_correlations,
    overall_rmse_mae,
)


IDENTITY_COLUMNS = [
    "sample_id",
    "fold",
    "complex",
    "mutstr",
    "num_muts",
    "ddG",
]
METRICS = [
    "overall_pearson",
    "overall_spearman",
    "rmse",
    "mae",
    "rmse_raw",
    "mae_raw",
    "auroc",
]
MAXIMIZE = {"overall_pearson", "overall_spearman", "auroc"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(path, mode):
    frame = pd.read_csv(path)
    required = set(IDENTITY_COLUMNS + ["method", "ddG_pred"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample_id values")
    if not np.isfinite(frame[["ddG", "ddG_pred"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite targets or predictions")
    if mode == "single":
        frame = frame.query("num_muts == 1")
    elif mode == "multiple":
        frame = frame.query("num_muts > 1")
    elif mode != "all":
        raise ValueError("mode must be all, single, or multiple")
    if frame.empty:
        raise ValueError(f"{path} has no rows for mode={mode}")
    return frame.sort_values("sample_id").reset_index(drop=True)


def metric_values(frame):
    values = {}
    values.update(overall_correlations(frame))
    values.update(overall_rmse_mae(frame))
    values.update(overall_auroc(frame))
    return {name: float(values[name]) for name in METRICS}


def align_reference(candidate, reference):
    if len(candidate) != len(reference):
        raise ValueError("Candidate and reference have different row counts")
    for column in IDENTITY_COLUMNS:
        left = candidate[column].to_numpy()
        right = reference[column].to_numpy()
        if column == "ddG":
            matches = np.allclose(left, right, rtol=0.0, atol=1e-6)
        else:
            matches = np.array_equal(left, right)
        if not matches:
            raise ValueError(f"Prediction identity mismatch in column {column}")
    aligned = candidate.copy()
    aligned["reference_ddG_pred"] = reference["ddG_pred"].to_numpy()
    return aligned


def stratified_cluster_indices(frame, rng):
    sampled = []
    for _, fold_frame in frame.groupby("fold", sort=True):
        groups = {
            complex_id: rows.to_numpy(dtype=int)
            for complex_id, rows in fold_frame.groupby("complex").groups.items()
        }
        complex_ids = sorted(groups)
        draws = rng.integers(0, len(complex_ids), size=len(complex_ids))
        for draw in draws:
            sampled.extend(groups[complex_ids[int(draw)]])
    return np.asarray(sampled, dtype=int)


def percentile_summary(values, confidence):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"bootstrap_mean": None, "bootstrap_se": None, "ci_low": None, "ci_high": None}
    alpha = (1.0 - confidence) / 2.0
    return {
        "bootstrap_mean": float(values.mean()),
        "bootstrap_se": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "ci_low": float(np.quantile(values, alpha)),
        "ci_high": float(np.quantile(values, 1.0 - alpha)),
    }


def bootstrap(frame, replicates, seed, confidence, paired=False):
    rng = np.random.default_rng(seed)
    candidate_values = {metric: [] for metric in METRICS}
    reference_values = {metric: [] for metric in METRICS}
    favorable_deltas = {metric: [] for metric in METRICS}

    for _ in range(replicates):
        indices = stratified_cluster_indices(frame, rng)
        sampled = frame.iloc[indices].copy()
        candidate = metric_values(sampled)
        for metric in METRICS:
            candidate_values[metric].append(candidate[metric])

        if paired:
            sampled["ddG_pred"] = sampled["reference_ddG_pred"]
            reference = metric_values(sampled)
            for metric in METRICS:
                reference_values[metric].append(reference[metric])
                if metric in MAXIMIZE:
                    delta = candidate[metric] - reference[metric]
                else:
                    delta = reference[metric] - candidate[metric]
                favorable_deltas[metric].append(delta)

    candidate_summary = {
        metric: percentile_summary(candidate_values[metric], confidence)
        for metric in METRICS
    }
    if not paired:
        return candidate_summary, None, None
    reference_summary = {
        metric: percentile_summary(reference_values[metric], confidence)
        for metric in METRICS
    }
    delta_summary = {
        metric: percentile_summary(favorable_deltas[metric], confidence)
        for metric in METRICS
    }
    return candidate_summary, reference_summary, delta_summary


def fold_metrics(frame):
    rows = []
    for fold, fold_frame in frame.groupby("fold", sort=True):
        rows.append({"fold": int(fold), **metric_values(fold_frame)})
    return pd.DataFrame(rows)


def fold_summary(frame):
    rows = []
    for metric in METRICS:
        values = frame[metric].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "fold_mean": float(values.mean()),
                "fold_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "fold_min": float(values.min()),
                "fold_max": float(values.max()),
                "num_folds": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(path, report):
    point = report["candidate"]["point"]
    intervals = report["candidate"]["bootstrap"]
    lines = [
        "# Table 3 Uncertainty Report",
        "",
        f"Method: `{report['candidate']['method']}`  ",
        f"Subset: `{report['mode']}` ({report['num_rows']} rows, {report['num_complexes']} complexes)  ",
        (
            f"Bootstrap: {report['bootstrap']['replicates']} complex-cluster replicates, "
            f"stratified by outer fold, seed {report['bootstrap']['seed']}."
        ),
        "",
        "| Metric | Point estimate | 95% cluster-bootstrap CI |",
        "|---|---:|---:|",
    ]
    for metric in METRICS:
        interval = intervals[metric]
        lines.append(
            f"| {metric} | {point[metric]:.4f} | "
            f"[{interval['ci_low']:.4f}, {interval['ci_high']:.4f}] |"
        )

    if report.get("reference"):
        lines.extend(
            [
                "",
                f"Reference: `{report['reference']['method']}`.",
                "Positive paired deltas favor the candidate for every metric.",
                "",
                "| Metric | Favorable point delta | 95% paired cluster-bootstrap CI |",
                "|---|---:|---:|",
            ]
        )
        for metric in METRICS:
            delta = report["paired_favorable_delta"][metric]
            lines.append(
                f"| {metric} | {delta['point']:.4f} | "
                f"[{delta['ci_low']:.4f}, {delta['ci_high']:.4f}] |"
            )
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mode", choices=["all", "single", "multiple"], default="single")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if os.path.exists(args.outdir) and os.listdir(args.outdir):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)

    candidate = load_predictions(args.predictions, args.mode)
    candidate_method = str(candidate["method"].iloc[0])
    candidate_point = metric_values(candidate)
    candidate_folds = fold_metrics(candidate)
    candidate_fold_summary = fold_summary(candidate_folds)

    reference = None
    reference_point = None
    reference_method = None
    bootstrap_frame = candidate
    if args.reference:
        reference = load_predictions(args.reference, args.mode)
        reference_method = str(reference["method"].iloc[0])
        bootstrap_frame = align_reference(candidate, reference)
        reference_point = metric_values(reference)

    candidate_bootstrap, reference_bootstrap, delta_bootstrap = bootstrap(
        bootstrap_frame,
        args.bootstrap_replicates,
        args.seed,
        args.confidence,
        paired=reference is not None,
    )

    report = {
        "schema_version": 1,
        "mode": args.mode,
        "num_rows": int(len(candidate)),
        "num_complexes": int(candidate["complex"].nunique()),
        "num_folds": int(candidate["fold"].nunique()),
        "candidate": {
            "method": candidate_method,
            "predictions": args.predictions,
            "sha256": sha256_file(args.predictions),
            "point": candidate_point,
            "bootstrap": candidate_bootstrap,
        },
        "bootstrap": {
            "unit": "complex",
            "stratification": "outer fold",
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.seed),
            "confidence": float(args.confidence),
            "interval": "percentile",
        },
    }
    if reference is not None:
        favorable_point = {}
        for metric in METRICS:
            if metric in MAXIMIZE:
                delta = candidate_point[metric] - reference_point[metric]
            else:
                delta = reference_point[metric] - candidate_point[metric]
            favorable_point[metric] = float(delta)
        report["reference"] = {
            "method": reference_method,
            "predictions": args.reference,
            "sha256": sha256_file(args.reference),
            "point": reference_point,
            "bootstrap": reference_bootstrap,
        }
        report["paired_favorable_delta"] = {
            metric: {"point": favorable_point[metric], **delta_bootstrap[metric]}
            for metric in METRICS
        }

    candidate_folds.to_csv(os.path.join(args.outdir, "fold_metrics.csv"), index=False)
    candidate_fold_summary.to_csv(
        os.path.join(args.outdir, "fold_variation.csv"), index=False
    )
    with open(os.path.join(args.outdir, "uncertainty.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(os.path.join(args.outdir, "uncertainty.md"), report)
    print(f"RUN_DIR={args.outdir}")


if __name__ == "__main__":
    main()
