#!/usr/bin/env python
"""Evaluate predicted nanobody PDBs against record-aligned native coordinates."""

import argparse
import csv
import json
import os

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.SVDSuperimposer import SVDSuperimposer


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
BACKBONE = ("N", "CA", "C", "O")


def load_subset(path):
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                records.append(int(line.split()[0]))
    return records


def load_source(path, selected):
    selected = set(selected)
    entries = {}
    with open(path) as handle:
        for index, line in enumerate(handle):
            if index in selected:
                entries[index] = json.loads(line)
    return entries


def load_prediction_manifest(path):
    rows = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["source_line"]] = row
    return rows


def parse_pdb(path):
    residues = []
    residue_lookup = {}
    with open(path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21].strip() or "_"
            key = (chain, line[22:26].strip(), line[26].strip())
            residue_name = line[17:20].strip().upper()
            if residue_name not in THREE_TO_ONE:
                continue
            if key not in residue_lookup:
                residue_lookup[key] = len(residues)
                residues.append(
                    {"aa": THREE_TO_ONE[residue_name], "atoms": {}, "key": key}
                )
            atom_name = line[12:16].strip()
            alternate = line[16].strip()
            if alternate not in {"", "A"} or atom_name in residues[residue_lookup[key]]["atoms"]:
                continue
            residues[residue_lookup[key]]["atoms"][atom_name] = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
    if not residues:
        raise ValueError("PDB contains no standard ATOM residues")
    return residues


def sequence_mapping(native_sequence, predicted_sequence):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(native_sequence, predicted_sequence)[0]
    mapping = {}
    for (native_begin, native_end), (predicted_begin, predicted_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        length = min(native_end - native_begin, predicted_end - predicted_begin)
        for offset in range(length):
            mapping[int(native_begin + offset)] = int(predicted_begin + offset)
    identity = sum(
        native_sequence[native] == predicted_sequence[predicted]
        for native, predicted in mapping.items()
    ) / max(1, len(mapping))
    return mapping, identity


def coordinate_pairs(entry, predicted, mapping, native_indices, atoms):
    native_values = []
    predicted_values = []
    for native_index in native_indices:
        predicted_index = mapping.get(native_index)
        if predicted_index is None:
            continue
        for atom_index, atom_name in enumerate(BACKBONE):
            if atom_name not in atoms:
                continue
            native_coord = np.asarray(
                entry["antibody_coords"][native_index][atom_index], dtype=np.float64
            )
            predicted_coord = predicted[predicted_index]["atoms"].get(atom_name)
            if predicted_coord is None or np.linalg.norm(native_coord) <= 1e-6:
                continue
            native_values.append(native_coord)
            predicted_values.append(predicted_coord)
    return np.asarray(native_values), np.asarray(predicted_values)


def superimpose(reference, moving):
    if len(reference) < 3 or len(reference) != len(moving):
        raise ValueError("at least three paired atoms are required for superposition")
    superimposer = SVDSuperimposer()
    superimposer.set(reference, moving)
    superimposer.run()
    rotation, translation = superimposer.get_rotran()
    transformed = moving @ rotation + translation
    return transformed, rotation, translation


def rmsd(reference, moving):
    return float(np.sqrt(np.square(reference - moving).sum(axis=1).mean()))


def evaluate_record(entry, prediction_path):
    predicted = parse_pdb(prediction_path)
    predicted_sequence = "".join(residue["aa"] for residue in predicted)
    native_sequence = entry["antibody_seq"]
    mapping, identity = sequence_mapping(native_sequence, predicted_sequence)
    cdr3 = [index for index, tag in enumerate(entry["antibody_cdr"]) if tag == "3"]
    framework = [index for index, tag in enumerate(entry["antibody_cdr"]) if tag == "0"]
    if not cdr3:
        raise ValueError("native record has no CDR3 residues")

    fw_native, fw_predicted = coordinate_pairs(
        entry, predicted, mapping, framework, set(BACKBONE[:3])
    )
    cdr_native, cdr_predicted = coordinate_pairs(
        entry, predicted, mapping, cdr3, set(BACKBONE[:3])
    )
    cdr_ca_native, cdr_ca_predicted = coordinate_pairs(
        entry, predicted, mapping, cdr3, {"CA"}
    )
    _, rotation, translation = superimpose(fw_native, fw_predicted)
    framework_aligned = cdr_predicted @ rotation + translation
    framework_aligned_ca = cdr_ca_predicted @ rotation + translation
    local_aligned, _, _ = superimpose(cdr_native, cdr_predicted)
    local_aligned_ca, _, _ = superimpose(cdr_ca_native, cdr_ca_predicted)
    mapped_cdr3 = sum(index in mapping for index in cdr3)
    return {
        "native_length": len(native_sequence),
        "predicted_length": len(predicted_sequence),
        "mapped_residues": len(mapping),
        "sequence_identity": identity,
        "cdr3_residues": len(cdr3),
        "mapped_cdr3_residues": mapped_cdr3,
        "framework_alignment_atoms": len(fw_native),
        "cdr3_backbone_atoms": len(cdr_native),
        "framework_aligned_cdr3_backbone_rmsd": rmsd(cdr_native, framework_aligned),
        "framework_aligned_cdr3_ca_rmsd": rmsd(cdr_ca_native, framework_aligned_ca),
        "local_cdr3_backbone_rmsd": rmsd(cdr_native, local_aligned),
        "local_cdr3_ca_rmsd": rmsd(cdr_ca_native, local_aligned_ca),
    }


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        default="abgnn/finetune/exp2-hern/nanobody_clustered/test_data.jsonl",
    )
    parser.add_argument(
        "--restrict-records",
        default="baselines/nanobody_clustered/eval_subset_records_pretrain_exact_disjoint.txt",
    )
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke-test only; omit for the publishable full-subset evaluation",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    selected = load_subset(args.restrict_records)
    if args.limit is not None:
        selected = selected[: args.limit]
    entries = load_source(args.source_jsonl, selected)
    predictions = load_prediction_manifest(
        os.path.join(args.prediction_dir, "prediction_results.jsonl")
    )
    rows = []
    failures = []
    for source_line in selected:
        prediction = predictions.get(source_line)
        if prediction is None or prediction.get("status") != "ok":
            failures.append(
                {
                    "source_line": source_line,
                    "reason": prediction.get("error", "missing prediction")
                    if prediction
                    else "missing prediction",
                }
            )
            continue
        try:
            metrics = evaluate_record(entries[source_line], prediction["output_pdb"])
            rows.append(
                {
                    "source_line": source_line,
                    "pdb": entries[source_line]["pdb"].lower(),
                    **metrics,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "source_line": source_line,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )

    os.makedirs(args.output_dir, exist_ok=True)
    if rows:
        with open(os.path.join(args.output_dir, "per_record.tsv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    with open(os.path.join(args.output_dir, "failures.tsv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_line", "reason"), delimiter="\t")
        writer.writeheader()
        writer.writerows(failures)
    metric_names = (
        "framework_aligned_cdr3_backbone_rmsd",
        "framework_aligned_cdr3_ca_rmsd",
        "local_cdr3_backbone_rmsd",
        "local_cdr3_ca_rmsd",
    )
    summary = {
        "selected_records": len(selected),
        "evaluated_records": len(rows),
        "failed_records": len(failures),
        "metrics": {
            metric: summarize([row[metric] for row in rows])
            for metric in metric_names
        }
        if rows
        else {},
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
