#!/usr/bin/env python
"""Export frozen Lm-Nano representations for one official NbBench dataset.

Representations are deduplicated across train/validation/test.  Split files retain
the original CSV row indices and map each row to the shared representation table.
This is important for the interaction datasets, where the same VHH occurs with
many antigen variants.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time

import numpy as np
import torch

from lmnano_encoder import LmNanoEncoder


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {residue: index for index, residue in enumerate(AA_ALPHABET)}
SPLITS = ("train", "val", "test")
TASKS = {
    "VRClassification": {
        "kind": "residue_classification",
        "sequence": "seq",
        "target": "label",
        "classes": 4,
    },
    "CDRInfilling": {
        "kind": "masked_residue_classification",
        "sequence": "seq",
        "target": "label",
        "classes": len(AA_ALPHABET),
    },
    "thermo-seq": {
        "kind": "regression",
        "sequence": "seq",
        "target": "label",
    },
    "thermo-tm": {
        "kind": "regression",
        "sequence": "seq",
        "target": "label",
    },
    "vhh_affinity-seq": {
        "kind": "regression",
        "sequence": "seq",
        "target": "score",
    },
    "vhh_affinity-score": {
        "kind": "regression",
        "sequence": "seq",
        "target": "score",
    },
    "polyreaction": {
        "kind": "classification",
        "sequence": "seq",
        "target": "label",
        "classes": 2,
    },
    "nanobody_type": {
        "kind": "classification",
        "sequence": "seq",
        "target": "label",
        "classes": 5,
    },
    "SARS-CoV-2": {
        "kind": "interaction",
        "sequence": "VHH_sequence",
        "antigen": "Ag_sequence",
        "target": "label",
        "classes": 2,
    },
    "hIL6": {
        "kind": "interaction",
        "sequence": "VHH_sequence",
        "antigen": "Ag_sequence",
        "target": "label",
        "classes": 2,
    },
    "Paratope": {
        "kind": "paratope",
        "sequence": "seq_nanobody",
        "antigen": "seq_antigen",
        "target": "paratope",
        "classes": 2,
    },
}


def normalize_sequence(value):
    return "".join(value.split()).upper()


def sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_exact_overlaps(path, dataset):
    overlaps = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"dataset", "split", "row_index", "pretrain_occurrences"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} does not contain the overlap audit columns")
        for row in reader:
            if row["dataset"] == dataset and int(row["pretrain_occurrences"]) > 0:
                overlaps.add((row["split"], int(row["row_index"])))
    return overlaps


def should_exclude_overlap(policy, split, overlap):
    if not overlap or policy == "keep":
        return False
    if policy == "exclude-all":
        return True
    return policy == "exclude-test" and split == "test"


def parse_target(task, sequence, raw_target):
    kind = task["kind"]
    if kind == "regression":
        return float(raw_target)
    if kind in {"classification", "interaction"}:
        target = int(raw_target)
        if not 0 <= target < task["classes"]:
            raise ValueError(f"class label {target} outside [0, {task['classes']})")
        return target
    if len(raw_target) != len(sequence):
        raise ValueError(
            f"residue target length {len(raw_target)} != sequence length {len(sequence)}"
        )
    if kind in {"residue_classification", "paratope"}:
        targets = [int(value) for value in raw_target]
        if any(value < 0 or value >= task["classes"] for value in targets):
            raise ValueError("residue label outside the configured class range")
        return targets
    positions = [index for index, residue in enumerate(sequence) if residue == "-"]
    if not positions:
        raise ValueError("CDR infilling row has no masked positions")
    try:
        return [AA_TO_INDEX[raw_target[index]] for index in positions]
    except KeyError as error:
        raise ValueError(f"unsupported CDR target residue {error.args[0]!r}") from error


def read_dataset(args, task, overlaps):
    split_rows = {}
    excluded_rows = []
    unique_sequences = []
    sequence_to_index = {}
    unique_antigens = []
    antigen_to_index = {}
    unknown_total = 0

    for split in SPLITS:
        path = os.path.join(args.data_dir, args.dataset, f"{split}.csv")
        rows = []
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            required = {task["sequence"], task["target"]}
            if "antigen" in task:
                required.add(task["antigen"])
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path} is missing columns {sorted(required)}")
            for row_index, row in enumerate(reader):
                sequence = normalize_sequence(row[task["sequence"]])
                overlap = (split, row_index) in overlaps
                reason = None
                if sequence == "SEQUENCE":
                    reason = "embedded_header_row"
                elif not sequence:
                    reason = "empty_sequence"
                elif len(sequence) > args.max_residues:
                    reason = "sequence_too_long"
                elif should_exclude_overlap(args.overlap_policy, split, overlap):
                    reason = "pretraining_exact_overlap"
                if reason is not None:
                    excluded_rows.append((split, row_index, reason))
                    continue

                try:
                    target = parse_target(task, sequence, row[task["target"]])
                except (TypeError, ValueError) as error:
                    if args.invalid_policy == "error":
                        raise ValueError(
                            f"invalid {args.dataset}/{split} row {row_index}: {error}"
                        ) from error
                    excluded_rows.append((split, row_index, f"invalid_target:{error}"))
                    continue

                unknown_count = sum(
                    residue not in AA_TO_INDEX and residue != "-" for residue in sequence
                )
                unknown_total += unknown_count
                if sequence not in sequence_to_index:
                    sequence_to_index[sequence] = len(unique_sequences)
                    unique_sequences.append(sequence)

                antigen_index = None
                if "antigen" in task:
                    antigen = normalize_sequence(row[task["antigen"]])
                    if not antigen:
                        if args.invalid_policy == "error":
                            raise ValueError(
                                f"empty antigen in {args.dataset}/{split} row {row_index}"
                            )
                        excluded_rows.append((split, row_index, "empty_antigen"))
                        continue
                    if antigen not in antigen_to_index:
                        antigen_to_index[antigen] = len(unique_antigens)
                        unique_antigens.append(antigen)
                    antigen_index = antigen_to_index[antigen]

                rows.append(
                    {
                        "row_index": row_index,
                        "sequence_index": sequence_to_index[sequence],
                        "target": target,
                        "overlap": overlap,
                        "unknown_count": unknown_count,
                        "antigen_index": antigen_index,
                        "target_positions": (
                            [
                                index
                                for index, residue in enumerate(sequence)
                                if residue == "-"
                            ]
                            if task["kind"] == "masked_residue_classification"
                            else None
                        ),
                    }
                )
                if args.limit_per_split and len(rows) >= args.limit_per_split:
                    break
        split_rows[split] = rows

    return {
        "splits": split_rows,
        "unique_sequences": unique_sequences,
        "unique_antigens": unique_antigens,
        "excluded_rows": excluded_rows,
        "unknown_total": unknown_total,
    }


def selector_for_sequence(kind, sequence):
    if kind == "masked_residue_classification":
        return [index for index, residue in enumerate(sequence) if residue == "-"]
    return range(len(sequence))


def atomic_memmap(path, dtype, shape):
    temporary = path + ".tmp"
    if os.path.exists(temporary):
        os.remove(temporary)
    return temporary, np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)


def finish_memmap(temporary, array, destination):
    array.flush()
    del array
    os.replace(temporary, destination)


def export_representations(args, task, dataset, encoder, output_dir):
    sequences = dataset["unique_sequences"]
    hidden_size = int(encoder.args.encoder_embed_dim)
    residue_level = task["kind"] in {
        "residue_classification",
        "masked_residue_classification",
        "paratope",
    }
    if residue_level:
        lengths = np.asarray(
            [len(selector_for_sequence(task["kind"], sequence)) for sequence in sequences],
            dtype=np.int64,
        )
        offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        shape = (int(offsets[-1]), hidden_size)
        np.save(os.path.join(output_dir, "representation_offsets.npy"), offsets)
    else:
        offsets = None
        shape = (len(sequences), hidden_size)

    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    destination = os.path.join(output_dir, "representations.npy")
    temporary, representations = atomic_memmap(destination, np_dtype, shape)
    started = time.time()
    for begin in range(0, len(sequences), args.batch_size):
        batch = sequences[begin : begin + args.batch_size]
        with torch.no_grad():
            if residue_level:
                features, _ = encoder.residue_embeddings(batch)
                for local_index, sequence in enumerate(batch):
                    sequence_index = begin + local_index
                    selector = selector_for_sequence(task["kind"], sequence)
                    values = features[local_index, selector]
                    representations[offsets[sequence_index] : offsets[sequence_index + 1]] = (
                        values.float().cpu().numpy().astype(np_dtype, copy=False)
                    )
            else:
                if args.pooling == "cls":
                    values = encoder.cls_embeddings(batch)
                else:
                    values = encoder.mean_embeddings(batch)
                representations[begin : begin + len(batch)] = (
                    values.float().cpu().numpy().astype(np_dtype, copy=False)
                )
        if begin == 0 or (begin // args.batch_size + 1) % args.log_every == 0:
            print(
                f"encoded {min(begin + len(batch), len(sequences))}/{len(sequences)} "
                f"unique VHHs in {time.time() - started:.1f}s",
                flush=True,
            )
    finish_memmap(temporary, representations, destination)
    return shape


def export_antigen_means(args, dataset, output_dir):
    antigens = dataset["unique_antigens"]
    if not antigens or args.skip_antigen_means:
        return None
    source_path = os.path.join(args.data_dir, args.dataset, "antigen_embeddings.pt")
    source = torch.load(source_path, map_location="cpu")
    missing = [antigen for antigen in antigens if antigen not in source]
    if missing:
        raise KeyError(f"{len(missing)} antigen sequences missing from {source_path}")

    first = source[antigens[0]].squeeze(0)
    hidden_size = int(first.shape[-1])
    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    destination = os.path.join(output_dir, "antigen_means.npy")
    temporary, means = atomic_memmap(
        destination, np_dtype, (len(antigens), hidden_size)
    )
    for index, antigen in enumerate(antigens):
        embedding = source[antigen]
        while embedding.ndim > 2 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        if embedding.ndim != 2 or embedding.shape[-1] != hidden_size:
            raise ValueError(
                f"unexpected antigen embedding shape {tuple(embedding.shape)} at index {index}"
            )
        means[index] = embedding.float().mean(dim=0).numpy().astype(np_dtype, copy=False)
        if index == 0 or (index + 1) % 100 == 0:
            print(f"pooled {index + 1}/{len(antigens)} antigens", flush=True)
    finish_memmap(temporary, means, destination)
    return (len(antigens), hidden_size)


def write_lines(path, values):
    with open(path, "w") as handle:
        for value in values:
            handle.write(value + "\n")


def write_split(output_dir, split, rows, task):
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    np.save(
        os.path.join(split_dir, "row_indices.npy"),
        np.asarray([row["row_index"] for row in rows], dtype=np.int64),
    )
    np.save(
        os.path.join(split_dir, "sequence_index.npy"),
        np.asarray([row["sequence_index"] for row in rows], dtype=np.int64),
    )
    np.save(
        os.path.join(split_dir, "pretrain_exact_overlap.npy"),
        np.asarray([row["overlap"] for row in rows], dtype=np.bool_),
    )
    np.save(
        os.path.join(split_dir, "unknown_residue_count.npy"),
        np.asarray([row["unknown_count"] for row in rows], dtype=np.int16),
    )
    if "antigen" in task:
        np.save(
            os.path.join(split_dir, "antigen_index.npy"),
            np.asarray([row["antigen_index"] for row in rows], dtype=np.int64),
        )

    residue_level = task["kind"] in {
        "residue_classification",
        "masked_residue_classification",
        "paratope",
    }
    if residue_level:
        lengths = np.asarray([len(row["target"]) for row in rows], dtype=np.int64)
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        targets = np.empty(int(offsets[-1]), dtype=np.int64)
        for index, row in enumerate(rows):
            targets[offsets[index] : offsets[index + 1]] = row["target"]
        np.save(os.path.join(split_dir, "label_offsets.npy"), offsets)
        if task["kind"] == "masked_residue_classification":
            positions = np.empty(int(offsets[-1]), dtype=np.int16)
            for index, row in enumerate(rows):
                positions[offsets[index] : offsets[index + 1]] = row[
                    "target_positions"
                ]
            np.save(os.path.join(split_dir, "target_positions.npy"), positions)
    elif task["kind"] == "regression":
        targets = np.asarray([row["target"] for row in rows], dtype=np.float32)
    else:
        targets = np.asarray([row["target"] for row in rows], dtype=np.int64)
    np.save(os.path.join(split_dir, "labels.npy"), targets)
    return {
        "rows": len(rows),
        "targets": len(targets),
        "pretrain_exact_overlap_rows": sum(row["overlap"] for row in rows),
        "rows_with_unknown_residues": sum(row["unknown_count"] > 0 for row in rows),
    }


def write_metadata(args, task, dataset, output_dir, representation_shape, antigen_shape):
    write_lines(os.path.join(output_dir, "unique_sequences.txt"), dataset["unique_sequences"])
    if dataset["unique_antigens"]:
        write_lines(os.path.join(output_dir, "antigen_sequences.txt"), dataset["unique_antigens"])
    split_summary = {
        split: write_split(output_dir, split, dataset["splits"][split], task)
        for split in SPLITS
    }
    with open(os.path.join(output_dir, "excluded_rows.tsv"), "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("split", "row_index", "reason"))
        writer.writerows(dataset["excluded_rows"])

    metadata = {
        "dataset": args.dataset,
        "task": task,
        "data_dir": os.path.abspath(args.data_dir),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "overlap_manifest": os.path.abspath(args.overlap_manifest),
        "overlap_policy": args.overlap_policy,
        "invalid_policy": args.invalid_policy,
        "dtype": args.dtype,
        "sequence_pooling": args.pooling,
        "representation_shape": representation_shape,
        "antigen_mean_shape": antigen_shape,
        "unique_sequences": len(dataset["unique_sequences"]),
        "unique_antigens": len(dataset["unique_antigens"]),
        "unknown_residue_occurrences": dataset["unknown_total"],
        "excluded_rows": len(dataset["excluded_rows"]),
        "limit_per_split": args.limit_per_split,
        "split_summary": split_summary,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(TASKS))
    parser.add_argument("--data-dir", default="baselines/nbbench/data")
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/pretrained/"
            "fixed_RoPE_abbert_warmup10000_lr0.0006_maxsen128_upfreq4_"
            "samplenone_clip0.0/checkpoint18.pt"
        ),
    )
    parser.add_argument(
        "--overlap-manifest",
        default="baselines/nbbench/overlap_audit/exact_overlap_positive_rows.tsv",
    )
    parser.add_argument(
        "--overlap-policy",
        choices=("keep", "exclude-test", "exclude-all"),
        default="keep",
    )
    parser.add_argument(
        "--invalid-policy", choices=("skip", "error"), default="error"
    )
    parser.add_argument("--output-dir", default="baselines/nbbench/lmnano_checkpoint18")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-residues", type=int, default=255)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--skip-antigen-means", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        parser.error("--limit-per-split must be positive")
    task = TASKS[args.dataset]
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    output_dir = os.path.join(args.output_dir, args.dataset)
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        if not args.overwrite:
            raise FileExistsError(f"non-empty output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    overlaps = load_exact_overlaps(args.overlap_manifest, args.dataset)
    dataset = read_dataset(args, task, overlaps)
    print(
        f"{args.dataset}: unique VHHs={len(dataset['unique_sequences'])}, "
        f"rows={sum(map(len, dataset['splits'].values()))}, "
        f"excluded={len(dataset['excluded_rows'])}",
        flush=True,
    )
    encoder = LmNanoEncoder(args.checkpoint, device=args.device, unknown_policy="unk")
    if args.max_residues > encoder.max_residues:
        raise ValueError(
            f"requested max residues {args.max_residues} exceeds checkpoint limit "
            f"{encoder.max_residues}"
        )
    representation_shape = export_representations(
        args, task, dataset, encoder, output_dir
    )
    antigen_shape = export_antigen_means(args, dataset, output_dir)
    metadata = write_metadata(
        args, task, dataset, output_dir, representation_shape, antigen_shape
    )
    print(json.dumps(metadata["split_summary"], indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
