#!/usr/bin/env python3
"""Fail unless a Table 3 metrics file meets every manuscript threshold."""

import argparse
import json
import os

import pandas as pd


DEFAULT_THRESHOLDS = {
    "overall_pearson": ("min", 0.6365),
    "overall_spearman": ("min", 0.5344),
    "rmse": ("max", 1.3652),
    "mae": ("max", 0.9617),
    "auroc": ("min", 0.7523),
}


def load_thresholds(path):
    """Load metric thresholds from a versioned gate JSON file."""
    with open(path) as handle:
        payload = json.load(handle)
    raw_thresholds = payload.get("thresholds", payload)
    thresholds = {}
    for metric, spec in raw_thresholds.items():
        if not isinstance(spec, dict):
            raise ValueError("Threshold for %s must be an object" % metric)
        direction = spec.get("direction")
        value = spec.get("value", spec.get("threshold"))
        if direction not in {"min", "max"}:
            raise ValueError("Threshold direction for %s must be min or max" % metric)
        if value is None:
            raise ValueError("Threshold for %s does not contain a value" % metric)
        thresholds[metric] = (direction, float(value))
    if not thresholds:
        raise ValueError("Threshold file is empty: %s" % path)
    return thresholds


def check_metrics(metrics_path, mode="single", thresholds=None):
    thresholds = thresholds or DEFAULT_THRESHOLDS
    table = pd.read_csv(metrics_path)
    if "mode" not in table:
        raise ValueError("Metrics file does not contain a mode column")
    selected = table[table["mode"] == mode]
    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one %s row in %s, found %d"
            % (mode, metrics_path, len(selected))
        )
    row = selected.iloc[0]
    checks = {}
    for metric, (direction, threshold) in thresholds.items():
        if metric not in row:
            raise ValueError("Missing metric: %s" % metric)
        value = float(row[metric])
        passed = value >= threshold if direction == "min" else value <= threshold
        checks[metric] = {
            "value": value,
            "direction": direction,
            "threshold": threshold,
            "passed": bool(passed),
        }
    return {
        "schema_version": 1,
        "metrics_path": os.path.abspath(metrics_path),
        "mode": mode,
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics")
    parser.add_argument("--mode", default="single")
    parser.add_argument("--thresholds-json", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    thresholds = load_thresholds(args.thresholds_json) if args.thresholds_json else None
    result = check_metrics(args.metrics, args.mode, thresholds=thresholds)
    if args.thresholds_json:
        result["thresholds_path"] = os.path.abspath(args.thresholds_json)
    for metric, item in result["checks"].items():
        operator = ">=" if item["direction"] == "min" else "<="
        print(
            "%-18s %.4f %s %.4f  %s"
            % (
                metric,
                item["value"],
                operator,
                item["threshold"],
                "PASS" if item["passed"] else "FAIL",
            )
        )
    print("TABLE3_ACCEPTANCE=%s" % ("PASS" if result["passed"] else "FAIL"))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
