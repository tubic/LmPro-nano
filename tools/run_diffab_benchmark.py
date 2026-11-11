#!/usr/bin/env python3
"""Run DiffAb fix-backbone CDR-H3 sampling on the common record-level split.

The adapter constructs DiffAb tensors directly from LmPro-Nano's atom14 JSONL, which
avoids antibody-chain renumbering ambiguity and supports VHHs without a light chain.
Only the same 20 closest antigen residues used by LmPro-Nano are exposed to the model.

Important: the official DiffAb checkpoint overlaps many benchmark PDBs.  Results from
that checkpoint are diagnostic only; use a checkpoint retrained on the clustered split
for a publishable comparison.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFAB_ROOT = os.path.join(ROOT, "third_party", "DiffAb")
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, DIFFAB_ROOT)

from diffab.models import get_model  # noqa: E402
from diffab.utils.data import PaddingCollate  # noqa: E402
from diffab.utils.inference import RemoveNative  # noqa: E402
from diffab.utils.protein import constants  # noqa: E402
from diffab.utils.train import recursive_to  # noqa: E402
from diffab.utils.transforms import MaskSingleCDR, MergeChains, PatchAroundAnchor  # noqa: E402
from export_proteinmpnn_benchmark import closest_antigen_residues  # noqa: E402
from record_ids import load_record_subset  # noqa: E402


INDEX_TO_AA = {index: residue for residue, index in constants.ressymb_to_resindex.items()}
ANTIGEN_CHAINS = [c for c in "ABCDEFGIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atom14_to_diffab(sequence, coordinates):
    if len(sequence) != len(coordinates):
        raise ValueError("sequence/coordinate length mismatch")
    xyz14 = torch.as_tensor(coordinates, dtype=torch.float32)
    if xyz14.ndim != 3 or tuple(xyz14.shape[1:]) != (14, 3):
        raise ValueError(f"expected (L, 14, 3) coordinates, got {tuple(xyz14.shape)}")
    xyz = torch.zeros((len(sequence), constants.max_num_heavyatoms, 3), dtype=torch.float32)
    xyz[:, :14] = xyz14
    mask = xyz.norm(dim=-1) > 1e-6
    aa = torch.tensor([int(constants.AA(residue)) for residue in sequence], dtype=torch.long)
    return aa, xyz, mask


def selected_segments(indices, antigen_coords):
    """Split selected antigen residues at source gaps or broken CA geometry."""
    if not indices:
        return []
    segments = [[indices[0]]]
    coords = torch.as_tensor(antigen_coords, dtype=torch.float32)
    for index in indices[1:]:
        previous = segments[-1][-1]
        ca_distance = torch.linalg.norm(coords[index, 1] - coords[previous, 1]).item()
        if index == previous + 1 and 0.0 < ca_distance <= 4.5:
            segments[-1].append(index)
        else:
            segments.append([index])
    return segments


def chain_data(sequence, coordinates, chain_ids, residue_numbers, cdr_tags=None):
    aa, xyz, atom_mask = atom14_to_diffab(sequence, coordinates)
    data = {
        "chain_id": chain_ids,
        "resseq": torch.tensor(residue_numbers, dtype=torch.long),
        "icode": [" "] * len(sequence),
        "res_nb": torch.tensor(residue_numbers, dtype=torch.long),
        "aa": aa,
        "pos_heavyatom": xyz,
        "mask_heavyatom": atom_mask,
    }
    if cdr_tags is not None:
        cdr_flag = torch.zeros(len(sequence), dtype=torch.long)
        cdr_flag[torch.tensor([tag == "1" for tag in cdr_tags])] = int(constants.CDR.H1)
        cdr_flag[torch.tensor([tag == "2" for tag in cdr_tags])] = int(constants.CDR.H2)
        cdr_flag[torch.tensor([tag == "3" for tag in cdr_tags])] = int(constants.CDR.H3)
        data["cdr_flag"] = cdr_flag
    return data


def build_structure(entry, source_line, epitope_size):
    antibody_seq = entry["antibody_seq"]
    antibody_cdr = entry["antibody_cdr"]
    cdr_indices = [i for i, tag in enumerate(antibody_cdr) if tag == "3"]
    if len(cdr_indices) <= 4 or cdr_indices != list(range(cdr_indices[0], cdr_indices[-1] + 1)):
        raise ValueError("invalid or non-contiguous CDR-H3")

    selected = closest_antigen_residues(
        entry["antibody_coords"], entry["antigen_coords"], cdr_indices, epitope_size
    )
    segments = selected_segments(selected, entry["antigen_coords"])
    if len(segments) > len(ANTIGEN_CHAINS):
        raise ValueError("too many disconnected antigen segments")

    antigen_indices = [index for segment in segments for index in segment]
    chain_ids = []
    for chain, segment in zip(ANTIGEN_CHAINS, segments):
        chain_ids.extend([chain] * len(segment))
    antigen_seq = "".join(entry["antigen_seq"][i] for i in antigen_indices)
    antigen_coords = [entry["antigen_coords"][i] for i in antigen_indices]
    antigen = chain_data(
        antigen_seq,
        antigen_coords,
        chain_ids,
        [i + 1 for i in antigen_indices],
    )
    name = f"line{source_line:06d}_{entry['pdb'].lower()}"
    structure = {
        "id": name,
        "heavy": chain_data(
            antibody_seq,
            entry["antibody_coords"],
            ["H"] * len(antibody_seq),
            list(range(1, len(antibody_seq) + 1)),
            antibody_cdr,
        ),
        "heavy_seqmap": None,
        "light": None,
        "light_seqmap": None,
        "antigen": antigen,
        "antigen_seqmap": None,
    }
    return structure, {
        "name": name,
        "source_line": source_line,
        "pdb": entry["pdb"].lower(),
        "native_cdr3": "".join(antibody_seq[i] for i in cdr_indices),
        "cdr3_start": cdr_indices[0],
        "cdr3_end": cdr_indices[-1] + 1,
        "epitope_indices": selected,
        "epitope_segments": [[segment[0], segment[-1]] for segment in segments],
    }


def prepare_data(structure):
    variant = MaskSingleCDR("H_CDR3", augmentation=False)(copy.deepcopy(structure))
    merged = MergeChains()(variant)
    cropped = PatchAroundAnchor()(copy.deepcopy(merged))
    cropped = RemoveNative(remove_structure=False, remove_sequence=True)(cropped)
    return cropped


def sample_record(model, data, num_samples, batch_size, device):
    collate = PaddingCollate(eight=False)
    candidates = []
    for start in range(0, num_samples, batch_size):
        count = min(batch_size, num_samples - start)
        batch = collate([data] * count)
        batch = recursive_to(batch, device)
        trajectory = model.sample(
            batch,
            sample_opt={"sample_structure": False, "sample_sequence": True},
        )
        generated = trajectory[0][2]
        generate_flag = batch["generate_flag"]
        for i in range(count):
            indices = generated[i][generate_flag[i]].tolist()
            candidates.append("".join(INDEX_TO_AA[int(index)] for index in indices))
        del trajectory, batch, generated
    return candidates


def summarize(record_files):
    matches = positions = candidates = 0
    per_record = []
    for path in record_files:
        with open(path) as handle:
            record = json.load(handle)
        native = record["native_cdr3"]
        record_matches = 0
        for prediction in record["candidates"]:
            if len(prediction) != len(native):
                raise ValueError(f"candidate length mismatch in {path}")
            record_matches += sum(a == b for a, b in zip(native, prediction))
            positions += len(native)
            candidates += 1
        matches += record_matches
        per_record.append(record_matches / (len(native) * len(record["candidates"])))
    return {
        "records": len(record_files),
        "candidates": candidates,
        "matched_positions": matches,
        "total_positions": positions,
        "aar": matches / positions,
        "macro_record_aar": sum(per_record) / len(per_record),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--restrict-records", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--epitope-size", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.num_samples <= 0 or args.batch_size <= 0 or args.epitope_size <= 0:
        parser.error("sample count, batch size, and epitope size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    selected = load_record_subset(args.restrict_records)
    checkpoint_sha = file_sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = get_model(checkpoint["config"].model).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    records_dir = os.path.join(args.output_dir, "records")
    os.makedirs(records_dir, exist_ok=True)
    completed = []
    with open(args.source_jsonl) as handle:
        for source_line, line in enumerate(handle):
            if source_line not in selected:
                continue
            if args.limit is not None and len(completed) >= args.limit:
                break
            entry = json.loads(line)
            structure, metadata = build_structure(entry, source_line, args.epitope_size)
            output_path = os.path.join(records_dir, metadata["name"] + ".json")
            if args.resume and os.path.exists(output_path):
                with open(output_path) as existing_handle:
                    existing = json.load(existing_handle)
                if (
                    existing.get("checkpoint_sha256") == checkpoint_sha
                    and existing.get("global_seed") == args.seed
                    and len(existing.get("candidates", [])) == args.num_samples
                ):
                    completed.append(output_path)
                    continue

            record_seed = args.seed + source_line
            seed_all(record_seed)
            data = prepare_data(structure)
            candidates = sample_record(
                model, data, args.num_samples, args.batch_size, args.device
            )
            native = metadata["native_cdr3"]
            if any(len(candidate) != len(native) for candidate in candidates):
                raise ValueError(f"DiffAb changed CDR-H3 length for {metadata['name']}")
            record = {
                **metadata,
                "checkpoint": os.path.abspath(args.checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "global_seed": args.seed,
                "record_seed": record_seed,
                "candidates": candidates,
            }
            temporary = output_path + ".tmp"
            with open(temporary, "w") as output_handle:
                json.dump(record, output_handle)
                output_handle.write("\n")
            os.replace(temporary, output_path)
            completed.append(output_path)
            print(f"[{len(completed)}] {metadata['name']} complete", flush=True)

    expected = min(len(selected), args.limit) if args.limit is not None else len(selected)
    if len(completed) != expected:
        raise SystemExit(f"expected {expected} records, completed {len(completed)}")
    metrics = summarize(completed)
    metrics.update(
        {
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "seed": args.seed,
            "samples_per_record": args.num_samples,
            "protocol": "unselected mean over fixed-backbone DiffAb samples",
        }
    )
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(
        f"DiffAb AAR = {100.0 * metrics['aar']:.2f}% "
        f"({metrics['matched_positions']}/{metrics['total_positions']} positions; "
        f"{metrics['records']} records x {args.num_samples} samples)"
    )
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
