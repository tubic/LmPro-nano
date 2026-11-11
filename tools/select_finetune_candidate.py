#!/usr/bin/env python3
"""Select a pretraining/fine-tuning candidate using validation evidence only."""

import argparse
import datetime as dt
import hashlib
import json
import math
import os


REQUIRED_KEYS = {
    "selection_metric",
    "seed",
    "protocol",
    "epoch",
    "aar",
    "rmsd",
    "ppl",
    "checkpoint",
    "checkpoint_sha256",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_selection(path):
    values = {}
    with open(path) as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{line_number}: expected key=value")
            key, value = line.split("=", 1)
            if not key or key in values:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate key {key!r}")
            values[key] = value
    missing = REQUIRED_KEYS - values.keys()
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    return values


def checked_metric(values, key, path):
    try:
        value = float(values[key])
    except ValueError as error:
        raise ValueError(f"{path}: {key} is not numeric: {values[key]!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{path}: {key} is not finite")
    return value


def load_candidate(label, selection_path, pretraining_checkpoint):
    values = read_selection(selection_path)
    finetuning_checkpoint = os.path.abspath(values["checkpoint"])
    pretraining_checkpoint = os.path.abspath(pretraining_checkpoint)
    for kind, path in (
        ("fine-tuning", finetuning_checkpoint),
        ("pretraining", pretraining_checkpoint),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing {kind} checkpoint for {label}: {path}")
    actual_finetuning_sha = sha256(finetuning_checkpoint)
    if actual_finetuning_sha != values["checkpoint_sha256"]:
        raise ValueError(
            f"fine-tuning checkpoint hash mismatch for {label}: "
            f"expected {values['checkpoint_sha256']}, got {actual_finetuning_sha}"
        )
    return {
        "label": label,
        "selection_file": os.path.abspath(selection_path),
        "selection_metric": values["selection_metric"],
        "seed": int(values["seed"]),
        "protocol": values["protocol"],
        "finetuning_epoch": int(values["epoch"]),
        "validation_aar": checked_metric(values, "aar", selection_path),
        "validation_rmsd": checked_metric(values, "rmsd", selection_path),
        "validation_ppl": checked_metric(values, "ppl", selection_path),
        "finetuning_checkpoint": finetuning_checkpoint,
        "finetuning_checkpoint_sha256": actual_finetuning_sha,
        "pretraining_checkpoint": pretraining_checkpoint,
        "pretraining_checkpoint_sha256": sha256(pretraining_checkpoint),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("LABEL", "SELECTION_FILE", "PRETRAINING_CHECKPOINT"),
        required=True,
        help="candidate label, validation selection file, and source pretraining checkpoint",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidates = [load_candidate(*specification) for specification in args.candidate]
    if len(candidates) < 2:
        parser.error("provide at least two --candidate entries")
    labels = [candidate["label"] for candidate in candidates]
    if len(labels) != len(set(labels)):
        parser.error("candidate labels must be unique")

    for field in ("selection_metric", "seed", "protocol"):
        observed = {candidate[field] for candidate in candidates}
        if len(observed) != 1:
            raise ValueError(f"candidates use different {field}: {sorted(observed)}")
    if candidates[0]["selection_metric"] != "validation_micro_aar":
        raise ValueError(
            "refusing non-validation selection metric: "
            f"{candidates[0]['selection_metric']}"
        )

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate["validation_aar"],
            candidate["validation_rmsd"],
            candidate["validation_ppl"],
            candidate["label"],
        ),
    )
    selected = ranked[0]
    output = {
        "selected_at": dt.datetime.now().astimezone().isoformat(),
        "selection_scope": "validation_only_test_not_read",
        "primary_metric": "maximum_validation_micro_aar",
        "tie_break": "minimum_validation_rmsd_then_minimum_validation_ppl_then_label",
        "seed": selected["seed"],
        "protocol": selected["protocol"],
        "selected": selected,
        "ranked_candidates": ranked,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = output_path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_path)
    print(
        f"selected {selected['label']} fine-tuning epoch "
        f"{selected['finetuning_epoch']} with validation AAR "
        f"{selected['validation_aar']:.6f}"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
