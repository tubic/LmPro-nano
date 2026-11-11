#!/usr/bin/env python3
"""Verify a frozen LmPro-Nano release bundle and its core record contracts."""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


EXPECTED_FILES = (
    "VERSION",
    "README.md",
    "MANIFEST.sha256",
    "ARTIFACTS.tsv",
    "EXTERNAL_ARTIFACTS.tsv",
    "inference.py",
    "data/seq_dict.txt",
    "data/tag_dict.txt",
    "checkpoints/release/lm_nano_pretrain_u33000.pt",
    "checkpoints/release/lmpro_nano_clustered_seed128_epoch51.pt",
    "checkpoints/release/lmpro_nano_antigen_disjoint_v2_seed128_epoch10.pt",
    "checkpoints/baselines/proteinmpnn_native_backbone_seed128_best.pt",
    "checkpoints/baselines/proteinmpnn_linear_init_seed128_best.pt",
    "checkpoints/baselines/diffab_clean_seed128_best.pt",
    "release_evidence/baselines/proteinmpnn_native_backbone_metrics.json",
    "release_evidence/baselines/proteinmpnn_native_backbone_per_record.tsv",
    "release_evidence/baselines/proteinmpnn_native_backbone_paired_bootstrap_top100.json",
    "release_evidence/baselines/proteinmpnn_linear_init_metrics.json",
    "release_evidence/baselines/proteinmpnn_linear_init_per_record.tsv",
    "release_evidence/baselines/proteinmpnn_linear_init_paired_bootstrap_top100.json",
    "release_evidence/lmpro/seed128_per_record.tsv",
    "release_evidence/lmpro/seed128_cluster_bootstrap_top100.json",
    "release_evidence/lmpro/seed128_structure_metrics.tsv",
    "release_evidence/lmpro/seed128_structure_metrics_summary.json",
    "release_evidence/lmpro/seed256_per_record.tsv",
    "release_evidence/lmpro/seed256_cluster_bootstrap_top100.json",
    "release_evidence/lmpro/seed256_structure_metrics.tsv",
    "release_evidence/lmpro/seed256_structure_metrics_summary.json",
    "release_evidence/lmpro/seed512_per_record.tsv",
    "release_evidence/lmpro/seed512_cluster_bootstrap_top100.json",
    "release_evidence/lmpro/seed512_structure_metrics.tsv",
    "release_evidence/lmpro/seed512_structure_metrics_summary.json",
    "baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt",
    "baselines/nanobody_antigen_disjoint_v2/test_records_pretrain_exact_disjoint_lmpro_loadable.txt",
)

EXPECTED_RECORD_COUNTS = {
    "baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt": 258,
    "baselines/nanobody_antigen_disjoint_v2/test_records_pretrain_exact_disjoint_lmpro_loadable.txt": 313,
}

PROTEINMPNN_EVIDENCE = {
    "native": {
        "metrics": "release_evidence/baselines/proteinmpnn_native_backbone_metrics.json",
        "per_record": "release_evidence/baselines/proteinmpnn_native_backbone_per_record.tsv",
        "bootstrap": "release_evidence/baselines/proteinmpnn_native_backbone_paired_bootstrap_top100.json",
    },
    "linear_init": {
        "metrics": "release_evidence/baselines/proteinmpnn_linear_init_metrics.json",
        "per_record": "release_evidence/baselines/proteinmpnn_linear_init_per_record.tsv",
        "bootstrap": "release_evidence/baselines/proteinmpnn_linear_init_paired_bootstrap_top100.json",
    },
}

LMPRO_EVIDENCE = {
    seed: {
        "per_record": f"release_evidence/lmpro/seed{seed}_per_record.tsv",
        "bootstrap": f"release_evidence/lmpro/seed{seed}_cluster_bootstrap_top100.json",
        "structure": f"release_evidence/lmpro/seed{seed}_structure_metrics.tsv",
        "structure_summary":
            f"release_evidence/lmpro/seed{seed}_structure_metrics_summary.json",
    }
    for seed in (128, 256, 512)
}

PUBLIC_CLAIM_FILES = (
    "README.md",
    "docs/REPRODUCIBILITY.md",
    "docs/MODIFICATIONS_VS_ABGNN.md",
    "docs/OPTIMIZATION_ALGORITHM.md",
)

PROHIBITED_PUBLIC_CLAIMS = (
    "framework-aligned cdr-h3 rmsd is 2.6133",
    "statistically neck-and-neck",
    "matched-protocol peer",
    "formal test in progress",
    "only method that jointly",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path):
    entries = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            expected, relative = raw.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid manifest line {lineno}: {raw!r}") from exc
        if len(expected) != 64 or Path(relative).is_absolute():
            raise ValueError(f"invalid manifest line {lineno}: {raw!r}")
        entries.append((expected, relative))
    return entries


def count_records(path):
    return sum(1 for line in path.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#"))


def verify_proteinmpnn_evidence(root, condition, paths):
    failures = []
    metrics_path = root / paths["metrics"]
    per_record_path = root / paths["per_record"]
    bootstrap_path = root / paths["bootstrap"]
    if not all(path.is_file() for path in (metrics_path, per_record_path, bootstrap_path)):
        return failures

    metrics = json.loads(metrics_path.read_text())
    if metrics.get("cdr_backbone_condition") != condition:
        failures.append(f"wrong ProteinMPNN condition in {paths['metrics']}")
    results = {row.get("topk"): row for row in metrics.get("results", [])}
    if set(results) != {1, 10, 100, 1000}:
        failures.append(f"wrong ProteinMPNN top-k set in {paths['metrics']}")
    for topk in (1, 10, 100, 1000):
        row = results.get(topk)
        if row is None:
            continue
        if row.get("records") != 258 or row.get("candidates_per_record") != 10000:
            failures.append(
                f"wrong ProteinMPNN record/candidate contract at top-{topk} in {paths['metrics']}"
            )
        if row.get("selected_designs") != 258 * topk:
            failures.append(
                f"wrong ProteinMPNN selected-design count at top-{topk} in {paths['metrics']}"
            )

    with per_record_path.open(newline="") as handle:
        per_record = list(csv.DictReader(handle, delimiter="\t"))
    if len(per_record) != 258 * 4:
        failures.append(f"wrong ProteinMPNN per-record row count in {paths['per_record']}")
    if {row.get("cdr_backbone_condition") for row in per_record} != {condition}:
        failures.append(f"wrong ProteinMPNN condition in {paths['per_record']}")
    keys = {(row.get("source_line"), row.get("topk")) for row in per_record}
    if len(keys) != len(per_record):
        failures.append(f"duplicate ProteinMPNN per-record keys in {paths['per_record']}")

    bootstrap = json.loads(bootstrap_path.read_text())
    if bootstrap.get("proteinmpnn_cdr_backbone_condition") != condition:
        failures.append(f"wrong ProteinMPNN condition in {paths['bootstrap']}")
    if bootstrap.get("records") != 258 or bootstrap.get("topk") != 100:
        failures.append(f"wrong ProteinMPNN bootstrap contract in {paths['bootstrap']}")
    if bootstrap.get("resampling_unit") != "pdb" or bootstrap.get("clusters") != 177:
        failures.append(f"ProteinMPNN bootstrap is not the 177-PDB cluster analysis in {paths['bootstrap']}")
    return failures


def verify_lmpro_evidence(root, seed, paths):
    failures = []
    per_record_path = root / paths["per_record"]
    bootstrap_path = root / paths["bootstrap"]
    structure_path = root / paths["structure"]
    structure_summary_path = root / paths["structure_summary"]
    if not all(
        path.is_file()
        for path in (per_record_path, bootstrap_path, structure_path, structure_summary_path)
    ):
        return failures

    with per_record_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 258 * 4:
        failures.append(f"wrong LmPro seed-{seed} per-record row count")
    required = {"source_line", "pdb", "topk", "succ", "tot", "local_cdr3_ca_rmsd"}
    if rows and not required.issubset(rows[0]):
        failures.append(f"missing explicit LmPro seed-{seed} per-record fields")
    if rows and "rmsd" in rows[0]:
        failures.append(f"ambiguous rmsd field remains in LmPro seed-{seed} evidence")
    keys = {(row.get("source_line"), row.get("topk")) for row in rows}
    if len(keys) != len(rows):
        failures.append(f"duplicate LmPro seed-{seed} source-line/top-k keys")
    top100 = [row for row in rows if row.get("topk") == "100"]
    if len(top100) != 258 or len({row.get("source_line") for row in top100}) != 258:
        failures.append(f"wrong LmPro seed-{seed} top-100 source-record contract")
    if len({row.get("pdb", "").lower() for row in top100}) != 177:
        failures.append(f"wrong LmPro seed-{seed} PDB-cluster count")

    bootstrap = json.loads(bootstrap_path.read_text())
    if (
        bootstrap.get("records") != 258
        or bootstrap.get("clusters") != 177
        or bootstrap.get("resampling_unit") != "pdb"
        or bootstrap.get("topk") != 100
    ):
        failures.append(f"wrong LmPro seed-{seed} cluster-bootstrap contract")

    with structure_path.open(newline="") as handle:
        structure = list(csv.DictReader(handle, delimiter="\t"))
    structure_fields = {
        "source_line",
        "pdb",
        "local_cdr3_ca_rmsd",
        "framework_anchored_cdr3_ca_rmsd",
        "framework_anchored_cdr3_backbone_rmsd",
    }
    if len(structure) != 258:
        failures.append(f"wrong LmPro seed-{seed} structure row count")
    if structure and not structure_fields.issubset(structure[0]):
        failures.append(f"missing LmPro seed-{seed} explicit structure metrics")
    if len({row.get("source_line") for row in structure}) != len(structure):
        failures.append(f"duplicate LmPro seed-{seed} structure source records")
    summary = json.loads(structure_summary_path.read_text())
    if summary.get("records") != 258:
        failures.append(f"wrong LmPro seed-{seed} structure summary record count")
    for metric in structure_fields - {"source_line", "pdb"}:
        reported = summary.get("metrics", {}).get(metric, {})
        values = [float(row[metric]) for row in structure] if structure else []
        if values and (
            abs(reported.get("mean", float("inf")) - statistics.fmean(values)) > 1e-9
            or abs(reported.get("median", float("inf")) - statistics.median(values)) > 1e-9
        ):
            failures.append(f"LmPro seed-{seed} {metric} summary does not match TSV")
    return failures


def verify_public_claims(root):
    failures = []
    for relative in PUBLIC_CLAIM_FILES:
        path = root / relative
        if not path.is_file():
            continue
        lowered = path.read_text().lower()
        for claim in PROHIBITED_PUBLIC_CLAIMS:
            if claim in lowered:
                failures.append(f"prohibited stale claim in {relative}: {claim!r}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-unit-tests", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    failures = []
    for relative in EXPECTED_FILES:
        if not (root / relative).is_file():
            failures.append(f"missing required file: {relative}")

    manifest_path = root / "MANIFEST.sha256"
    if manifest_path.is_file():
        try:
            entries = parse_manifest(manifest_path)
        except ValueError as exc:
            failures.append(str(exc))
            entries = []
        for expected, relative in entries:
            path = root / relative
            if not path.is_file():
                failures.append(f"manifest file missing: {relative}")
            elif sha256(path) != expected:
                failures.append(f"checksum mismatch: {relative}")

    for relative, expected in EXPECTED_RECORD_COUNTS.items():
        path = root / relative
        if path.is_file():
            observed = count_records(path)
            if observed != expected:
                failures.append(
                    f"record-count mismatch: {relative}: expected {expected}, got {observed}"
                )

    for condition, paths in PROTEINMPNN_EVIDENCE.items():
        failures.extend(verify_proteinmpnn_evidence(root, condition, paths))
    for seed, paths in LMPRO_EVIDENCE.items():
        failures.extend(verify_lmpro_evidence(root, seed, paths))
    failures.extend(verify_public_claims(root))

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(f"verified {len(entries)} checksummed files under {root}")
    print("record contracts: clustered=258, antigen-disjoint=313")

    if args.run_unit_tests:
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
