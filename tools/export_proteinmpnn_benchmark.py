#!/usr/bin/env python3
"""Export record-aligned CDR-H3 inputs for the official ProteinMPNN runner.

The source LmPro-Nano JSONL already contains atom14 coordinates, so using it directly
avoids ambiguous PDB-chain matching when one PDB contains several complexes.  Every
exported target has:

* chain H: the complete antibody sequence/backbone, with only CDR-H3 designable;
* fixed antigen chains: the same ``--epitope-size`` closest antigen residues used by
  LmPro-Nano (20 by default), split into contiguous source-sequence segments so that
  ProteinMPNN is never told that two non-adjacent residues share a peptide bond.

``--cdr-backbone native`` preserves the native CDR-H3 backbone and is an oracle-backbone
sequence-design condition. ``--cdr-backbone linear_init`` replaces those coordinates with
the same endpoint interpolation used to initialize LmPro-Nano, enabling a matched-input
control when used consistently for training, validation, and testing.

The output files are accepted directly by ``third_party/ProteinMPNN/protein_mpnn_run.py``.
Record identity is the zero-based source JSONL line number, consistent with
``tools/record_ids.py`` and the other clean-split evaluations.
"""

import argparse
import csv
import json
import math
import os
import string

import numpy as np
import torch


BACKBONE = ("N", "CA", "C", "O")
ANTIGEN_CHAIN_IDS = [c for c in string.ascii_uppercase + string.ascii_lowercase if c != "H"]


def load_record_subset(path):
    if path is None:
        return None
    records = set()
    with open(path) as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                records.add(int(line.split()[0]))
    if not records:
        raise ValueError(f"record subset is empty: {path}")
    return records


def closest_antigen_residues(antibody_coords, antigen_coords, cdr_indices, count):
    """Reproduce AntibodyComplexDataset's minimum-heavy-atom epitope selection."""
    paratope = torch.as_tensor(
        [antibody_coords[i] for i in cdr_indices], dtype=torch.float32
    )
    antigen = torch.as_tensor(antigen_coords, dtype=torch.float32)
    paratope_mask = paratope.norm(dim=-1) > 1e-6
    antigen_mask = antigen.norm(dim=-1) > 1e-6
    if not paratope_mask.any():
        raise ValueError("CDR-H3 has no valid atoms")

    # This is the same float32 reduction and torch.topk tie behaviour as
    # fairseq_models.modules.utils.full_square_dist(..., contact=True), followed by
    # ``amin(dim=-1).topk(largest=False)`` in AntibodyComplexDataset.
    delta = antigen[:, :, None, None, :] - paratope[None, None, :, :, :]
    distances = delta.square().sum(dim=-1)
    pair_mask = antigen_mask[:, :, None, None] & paratope_mask[None, None, :, :]
    distances = distances + 1e6 * (~pair_mask).to(distances.dtype)
    distances = distances.flatten(start_dim=1).amin(dim=-1)

    keep = min(count, len(antigen_coords))
    if keep == 0:
        raise ValueError("antigen is empty")
    selected = distances.topk(k=keep, largest=False).indices.sort().values
    return [int(i) for i in selected]


def contiguous_segments(indices):
    if not indices:
        return []
    segments = [[indices[0]]]
    for index in indices[1:]:
        if index == segments[-1][-1] + 1:
            segments[-1].append(index)
        else:
            segments.append([index])
    return segments


def backbone_dict(coords, chain_id):
    """Convert atom14 coordinates to ProteinMPNN's four-backbone-atom schema."""
    out = {}
    for atom_index, atom_name in enumerate(BACKBONE):
        values = []
        for residue in coords:
            xyz = [float(v) for v in residue[atom_index]]
            if np.linalg.norm(xyz) <= 1e-6:
                xyz = [math.nan, math.nan, math.nan]
            values.append(xyz)
        out[f"{atom_name}_chain_{chain_id}"] = values
    return out


def linearize_cdr_coordinates(antibody_coords, cdr_indices):
    """Return LmPro-Nano's endpoint-interpolated CDR coordinate initialization."""
    if cdr_indices[0] == 0 or cdr_indices[-1] + 1 >= len(antibody_coords):
        raise ValueError("CDR-H3 has no flanking framework residue")
    coordinates = torch.as_tensor(antibody_coords, dtype=torch.float32).clone()
    left = coordinates[cdr_indices[0] - 1]
    right = coordinates[cdr_indices[-1] + 1]
    span = len(cdr_indices) + 1
    offsets = (right - left).unsqueeze(0).expand(len(cdr_indices), -1, -1)
    initialized = left + torch.cumsum(offsets, dim=0) / span
    initialized[:, 4:] = initialized[:, 1].unsqueeze(1)
    coordinates[cdr_indices] = initialized
    return coordinates


def build_target(entry, source_line, epitope_size, cdr_backbone="native"):
    antibody_seq = entry["antibody_seq"]
    antibody_cdr = entry["antibody_cdr"]
    antibody_coords = entry["antibody_coords"]
    antigen_seq = entry["antigen_seq"]
    antigen_coords = entry["antigen_coords"]

    if not (len(antibody_seq) == len(antibody_cdr) == len(antibody_coords)):
        raise ValueError("antibody sequence, tags, and coordinates have different lengths")
    if len(antigen_seq) != len(antigen_coords):
        raise ValueError("antigen sequence and coordinates have different lengths")

    cdr_indices = [i for i, tag in enumerate(antibody_cdr) if tag == "3"]
    if len(cdr_indices) <= 4:
        raise ValueError("CDR-H3 has four or fewer residues")
    if cdr_indices != list(range(cdr_indices[0], cdr_indices[-1] + 1)):
        raise ValueError("CDR-H3 tags are not contiguous")

    selected = closest_antigen_residues(
        antibody_coords, antigen_coords, cdr_indices, epitope_size
    )
    segments = contiguous_segments(selected)
    if len(segments) > len(ANTIGEN_CHAIN_IDS):
        raise ValueError(f"epitope needs {len(segments)} chains; only {len(ANTIGEN_CHAIN_IDS)} available")

    if cdr_backbone == "native":
        exported_antibody_coords = antibody_coords
    elif cdr_backbone == "linear_init":
        exported_antibody_coords = linearize_cdr_coordinates(
            antibody_coords, cdr_indices
        )
    else:
        raise ValueError(f"unsupported CDR-H3 backbone condition: {cdr_backbone}")

    name = f"line{source_line:06d}_{entry['pdb'].lower()}"
    target = {
        "name": name,
        "seq_chain_H": antibody_seq,
        "coords_chain_H": backbone_dict(exported_antibody_coords, "H"),
    }
    visible_chains = []
    epitope_sequence = []
    for chain_id, segment in zip(ANTIGEN_CHAIN_IDS, segments):
        visible_chains.append(chain_id)
        sequence = "".join(antigen_seq[i] for i in segment)
        coords = [antigen_coords[i] for i in segment]
        target[f"seq_chain_{chain_id}"] = sequence
        target[f"coords_chain_{chain_id}"] = backbone_dict(coords, chain_id)
        epitope_sequence.append(sequence)

    target["num_of_chains"] = 1 + len(visible_chains)
    target["seq"] = antibody_seq + "".join(epitope_sequence)
    fixed_h = [i + 1 for i, tag in enumerate(antibody_cdr) if tag != "3"]
    manifest = {
        "name": name,
        "source_line": source_line,
        "pdb": entry["pdb"].lower(),
        "antibody_sequence": antibody_seq,
        "antibody_length": len(antibody_seq),
        "cdr3_start": cdr_indices[0],
        "cdr3_end": cdr_indices[-1] + 1,
        "cdr3_sequence": "".join(antibody_seq[i] for i in cdr_indices),
        "cdr_backbone_condition": cdr_backbone,
        "epitope_indices": ",".join(str(i) for i in selected),
        "epitope_segments": ",".join(f"{s[0]}-{s[-1]}" for s in segments),
        "epitope_sequence": "".join(epitope_sequence),
        "visible_chains": ",".join(visible_chains),
        "total_length": len(target["seq"]),
    }
    return target, (["H"], visible_chains), {"H": fixed_h}, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument(
        "--restrict-records",
        help="optional file of zero-based source JSONL line indices; comments are allowed",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epitope-size", type=int, default=20)
    parser.add_argument(
        "--cdr-backbone",
        choices=("native", "linear_init"),
        default="native",
        help="native is an oracle-backbone condition; linear_init matches LmPro initialization",
    )
    parser.add_argument("--limit", type=int, help="export only the first N selected records (smoke tests)")
    args = parser.parse_args()

    if args.epitope_size <= 0:
        parser.error("--epitope-size must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    selected_records = load_record_subset(args.restrict_records)
    os.makedirs(args.output_dir, exist_ok=True)
    parsed_path = os.path.join(args.output_dir, "parsed_pdbs.jsonl")
    chains_path = os.path.join(args.output_dir, "assigned_chains.jsonl")
    fixed_path = os.path.join(args.output_dir, "fixed_positions.jsonl")
    manifest_path = os.path.join(args.output_dir, "manifest.tsv")

    targets = []
    chain_assignments = {}
    fixed_positions = {}
    manifests = []
    failures = []
    with open(args.source_jsonl) as handle:
        for source_line, line in enumerate(handle):
            if selected_records is not None and source_line not in selected_records:
                continue
            if args.limit is not None and len(targets) >= args.limit:
                break
            try:
                target, chains, fixed, manifest = build_target(
                    json.loads(line), source_line, args.epitope_size, args.cdr_backbone
                )
            except (KeyError, TypeError, ValueError) as exc:
                failures.append((source_line, str(exc)))
                continue
            targets.append(target)
            chain_assignments[target["name"]] = chains
            fixed_positions[target["name"]] = fixed
            manifests.append(manifest)

    if failures:
        details = "; ".join(f"line {line}: {reason}" for line, reason in failures[:10])
        raise SystemExit(f"refusing a partial benchmark: {len(failures)} records failed ({details})")
    expected = len(selected_records) if selected_records is not None and args.limit is None else None
    if expected is not None and len(targets) != expected:
        raise SystemExit(f"selected {expected} records but exported {len(targets)}")
    if not targets:
        raise SystemExit("no records exported")

    with open(parsed_path, "w") as handle:
        for target in targets:
            handle.write(json.dumps(target, allow_nan=True) + "\n")
    with open(chains_path, "w") as handle:
        handle.write(json.dumps(chain_assignments) + "\n")
    with open(fixed_path, "w") as handle:
        handle.write(json.dumps(fixed_positions) + "\n")
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifests[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifests)

    print(f"exported {len(targets)} record-aligned targets")
    print(f"  parsed structures : {parsed_path}")
    print(f"  chain assignments : {chains_path}")
    print(f"  fixed positions   : {fixed_path}")
    print(f"  manifest          : {manifest_path}")


if __name__ == "__main__":
    main()
