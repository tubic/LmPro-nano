#!/usr/bin/env python3
"""Build the preregistered equal-weight Table 3 ensemble."""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from rde.utils.skempi import eval_skempi_three_modes


IDENTITY_COLUMNS = [
    "sample_id",
    "fold",
    "complex",
    "mutstr",
    "num_muts",
    "ddG",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fixed_equal_ensemble(paths, method=None, legacy_method=None):
    """Average fixed prediction files after strict identity alignment.

    The current API takes ``(paths, method)``.  The two-file positional form
    ``(first_path, second_path, method)`` is retained for older callers and
    tests so a protocol utility change cannot silently break downstream code.
    """
    if isinstance(paths, (str, os.PathLike)):
        if not isinstance(method, (str, os.PathLike)):
            raise TypeError("legacy ensemble calls require two prediction paths")
        input_paths = [os.fspath(paths), os.fspath(method)]
        method_name = legacy_method or "ProFE-fixed-equal-ensemble"
    else:
        input_paths = [os.fspath(path) for path in paths]
        method_name = method or "ProFE-fixed-equal-ensemble"

    if len(input_paths) < 2:
        raise ValueError("Need at least two prediction files to ensemble")

    tables = [pd.read_csv(path) for path in input_paths]
    for path, table in zip(input_paths, tables):
        missing = set(IDENTITY_COLUMNS + ["ddG_pred"]) - set(table.columns)
        if missing:
            raise ValueError("%s is missing columns: %s" % (path, sorted(missing)))
        if table["sample_id"].duplicated().any():
            raise ValueError("%s contains duplicate sample_id values" % path)
        if not np.isfinite(table["ddG_pred"].to_numpy(dtype=float)).all():
            raise ValueError("%s contains non-finite predictions" % path)

    tables = [table.sort_values("sample_id").reset_index(drop=True) for table in tables]
    for table in tables[1:]:
        if len(table) != len(tables[0]):
            raise ValueError("Prediction files have different row counts")
        for column in IDENTITY_COLUMNS:
            left = tables[0][column].to_numpy()
            right = table[column].to_numpy()
            if column == "ddG":
                matches = np.allclose(left, right, rtol=0.0, atol=1e-6)
            else:
                matches = np.array_equal(left, right)
            if not matches:
                raise ValueError("Prediction identity mismatch in column %s" % column)

    output = tables[0][IDENTITY_COLUMNS].copy()
    output["method"] = method_name
    output["ddG_pred"] = np.mean(
        [table["ddG_pred"].to_numpy() for table in tables], axis=0
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--method", default="ProFE-fixed-equal-ensemble")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        raise ValueError("Need at least two prediction files to ensemble")
    if os.path.exists(args.outdir) and os.listdir(args.outdir):
        raise FileExistsError("Refusing to overwrite non-empty %s" % args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    output = fixed_equal_ensemble(args.inputs, args.method)
    predictions_path = os.path.join(args.outdir, "outer_predictions.csv")
    metrics_path = os.path.join(args.outdir, "table3_metrics.csv")
    output.to_csv(predictions_path, index=False)
    eval_skempi_three_modes(output).to_csv(metrics_path, index=False)

    per_fold = []
    for fold in sorted(output["fold"].unique()):
        metrics = eval_skempi_three_modes(output.query("fold == @fold"))
        metrics["fold"] = int(fold)
        per_fold.append(metrics)
    pd.concat(per_fold, ignore_index=True).to_csv(
        os.path.join(args.outdir, "table3_metrics_per_fold.csv"), index=False
    )
    with open(os.path.join(args.outdir, "ensemble_manifest.json"), "w") as handle:
        json.dump(
            {
                "schema_version": 1,
                "method": args.method,
                "weights": [1.0 / len(args.inputs)] * len(args.inputs),
                "prediction_dependent_tuning": False,
                "inputs": [
                    {"path": os.path.abspath(path), "sha256": sha256_file(path)}
                    for path in args.inputs
                ],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print("RUN_DIR=%s" % args.outdir)


if __name__ == "__main__":
    main()
