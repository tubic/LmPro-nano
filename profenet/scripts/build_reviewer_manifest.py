#!/usr/bin/env python3
"""Build a machine-readable audit of the artifacts needed for revision."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reviewer_revision"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path, required=True):
    path = Path(path)
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "exists": False, "required": required}
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "exists": True,
        "required": required,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def parse_mdp(path):
    values = {}
    if not path.exists():
        return values
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        values[key] = value
    out = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "parameters": values,
    }
    try:
        dt_ps = float(values["dt"])
        nsteps = int(values["nsteps"])
        out["duration_ps"] = dt_ps * nsteps
        out["duration_ns"] = dt_ps * nsteps / 1000.0
    except (KeyError, TypeError, ValueError):
        pass
    return out


def run_git_status():
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"repository": False, "note": "No git metadata is present in the working tree."}
    return {
        "repository": True,
        "commit": result.stdout.strip(),
        "status": subprocess.run(
            ["git", "-C", str(ROOT), "status", "--short"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines(),
    }


def metrics_rows():
    rows = []
    for path in sorted((ROOT / "outputs" / "table3_runs").glob("*/table3_metrics.csv")):
        try:
            frame = pd.read_csv(path)
            row = frame.query("mode == 'single'").iloc[0].to_dict()
        except (FileNotFoundError, IndexError, KeyError, pd.errors.ParserError):
            continue
        rows.append(
            {
                "run": str(path.parent.relative_to(ROOT)),
                "method": row.get("method"),
                "overall_pearson": row.get("overall_pearson"),
                "overall_spearman": row.get("overall_spearman"),
                "rmse": row.get("rmse"),
                "mae": row.get("mae"),
                "auroc": row.get("auroc"),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(OUT / "reproducibility_manifest.json"),
        help="Manifest path (default: outputs/reviewer_revision/reproducibility_manifest.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tracked = [
        ROOT / "start.pdf",
        ROOT / "Reviewer 1 2 4.pdf",
        ROOT / "Reviewer 3.pdf",
        ROOT / "README.md",
        ROOT / "env.yml",
        ROOT / "requirements-reviewer.txt",
        ROOT / "screen_inner_dev.py",
        ROOT / "train_table3.py",
        ROOT / "ensemble_table3_predictions.py",
        ROOT / "summarize_table3_uncertainty.py",
        ROOT / "scripts" / "build_reviewer_manifest.py",
        ROOT / "scripts" / "reproduce_reviewer_table3.sh",
        ROOT / "docs" / "EVALUATION.md",
        ROOT / "rde" / "models" / "rde_2025.py",
        ROOT / "rde" / "utils" / "skempi.py",
        ROOT / "data" / "generated" / "pdbredo_skempi50_disjoint" / "audit.json",
        ROOT / "outputs" / "reviewer_revision" / "table3_uncertainty_profenet_v3_vs_rde" / "uncertainty.json",
        ROOT / "outputs" / "reviewer_revision" / "table3_uncertainty_fixed_ensemble_vs_rde" / "uncertainty.json",
        ROOT / "configs" / "train" / "profe_enhanced_table3.yml",
        ROOT / "configs" / "train" / "profe_enhanced_table3_v2_nested_control.yml",
        ROOT / "configs" / "train" / "table3_candidate_plan.json",
    ]
    packaged_md_root = ROOT / "external_audit" / "md_protocol"
    md_root = packaged_md_root if packaged_md_root.exists() else ROOT.parent / "gjxscript"
    mdp = [
        parse_mdp(md_root / "step4.0_minimization.mdp"),
        parse_mdp(md_root / "step4.1_equilibration.mdp"),
        parse_mdp(md_root / "step5_production.mdp"),
    ]
    missing_assets = [
        "Lm-Nano source/checkpoint and SabDab-Nano processed split",
        "NbBench data and task runner",
        "ProteinMPNN, Cmai, AlphaBind, GearBind, DiffAb, IgFold benchmark inputs/checkpoints",
        "18 production trajectories, topology files, MM-PBSA outputs, and DCCM/SASA/PCA tables",
        "editable manuscript source and supplementary figure files",
        "original MMseqs2 command line for the inherited PDB-REDO cluster file",
    ]
    report = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "git": run_git_status(),
        "tracked_artifacts": [artifact(path) for path in tracked],
        "md_protocol_audit": mdp,
        "md_submission_script": artifact(md_root / "submit_MD.sh", required=False),
        "table3_single_metrics": metrics_rows(),
        "known_missing_assets": missing_assets,
        "protocol_findings": [
            "The manuscript text says five-fold SKEMPI cross-validation, while the matched Table 3 reproduction uses the RDE-PPI seed-2022 three-fold complex split and single-mutation subset.",
            "The matched SKEMPI complex split has exact complex disjointness but PDB identifiers 2C5D and 3SE4 occur in more than one fold.",
            "The generated PDB-REDO split removes every source cluster containing a usable SKEMPI PDB and has zero exact chain/PDB overlap across train, validation, and test; the source MMseqs2 command is unavailable.",
            "The checked production MDP parameters correspond to 20 ns per production trajectory, not 200 ns; the reported trajectory files are absent.",
            "Protocol-corrected RMSE/MAE fit an affine calibration on evaluated labels; raw RMSE/MAE are reported separately.",
        ],
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
