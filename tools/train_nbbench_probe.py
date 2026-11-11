#!/usr/bin/env python
"""Train reproducible frozen-encoder probes from exported Lm-Nano features."""

import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import scipy.stats
import torch
from Bio.Align import substitution_matrices
from sklearn import metrics as sklearn_metrics
from torch import nn
from torch.utils.data import DataLoader, Dataset


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
RESIDUE_KINDS = {
    "residue_classification",
    "masked_residue_classification",
    "paratope",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rows_kept(overlaps, split, policy):
    if policy == "keep" or (policy == "exclude-test" and split != "test"):
        return np.ones(len(overlaps), dtype=np.bool_)
    return ~overlaps


class ProbeDataset(Dataset):
    """Map split rows onto shared VHH and antigen representation tables."""

    def __init__(
        self,
        root,
        split,
        task,
        overlap_policy,
        representations,
        antigens=None,
    ):
        self.root = root
        self.split = split
        self.task = task
        self.kind = task["kind"]
        self.representations = representations
        split_dir = os.path.join(root, split)
        sequence_index = np.load(os.path.join(split_dir, "sequence_index.npy"))
        row_indices = np.load(os.path.join(split_dir, "row_indices.npy"))
        overlaps = np.load(os.path.join(split_dir, "pretrain_exact_overlap.npy"))
        keep = rows_kept(overlaps, split, overlap_policy)
        self.original_row_indices = row_indices[keep]
        self.filtered_overlap_rows = int((~keep).sum())
        self.retained_overlap_rows = int(overlaps[keep].sum())

        antigen_index = None
        self.antigens = antigens
        if "antigen" in task:
            antigen_path = os.path.join(root, "antigen_means.npy")
            if not os.path.exists(antigen_path):
                raise FileNotFoundError(
                    f"{antigen_path} is required; export without --skip-antigen-means"
                )
            if self.antigens is None:
                raise ValueError("shared antigen table was not loaded")
            antigen_index = np.load(os.path.join(split_dir, "antigen_index.npy"))[keep]

        labels = np.load(os.path.join(split_dir, "labels.npy"), mmap_mode="r")
        self.row_offsets = None
        self.target_positions = None
        if self.kind in RESIDUE_KINDS:
            representation_offsets = np.load(
                os.path.join(root, "representation_offsets.npy")
            )
            label_offsets = np.load(os.path.join(split_dir, "label_offsets.npy"))
            selected_rows = np.flatnonzero(keep)
            lengths = label_offsets[selected_rows + 1] - label_offsets[selected_rows]
            feature_chunks = []
            label_chunks = []
            antigen_chunks = []
            position_chunks = []
            all_positions = None
            if self.kind == "masked_residue_classification":
                all_positions = np.load(
                    os.path.join(split_dir, "target_positions.npy"), mmap_mode="r"
                )
            for output_row, source_row in enumerate(selected_rows):
                sequence_id = int(sequence_index[source_row])
                feature_begin = int(representation_offsets[sequence_id])
                feature_end = int(representation_offsets[sequence_id + 1])
                label_begin = int(label_offsets[source_row])
                label_end = int(label_offsets[source_row + 1])
                if feature_end - feature_begin != label_end - label_begin:
                    raise ValueError(
                        f"representation/label length mismatch in {split} row "
                        f"{int(row_indices[source_row])}"
                    )
                feature_chunks.append(np.arange(feature_begin, feature_end, dtype=np.int64))
                label_chunks.append(np.asarray(labels[label_begin:label_end]))
                if antigen_index is not None:
                    antigen_chunks.append(
                        np.full(
                            label_end - label_begin,
                            antigen_index[output_row],
                            dtype=np.int64,
                        )
                    )
                if all_positions is not None:
                    position_chunks.append(
                        np.asarray(all_positions[label_begin:label_end], dtype=np.int16)
                    )
            self.feature_indices = concatenate_or_empty(
                feature_chunks, np.int64
            )
            self.labels = concatenate_or_empty(label_chunks, np.int64)
            self.antigen_indices = (
                concatenate_or_empty(antigen_chunks, np.int64)
                if antigen_index is not None
                else None
            )
            self.row_offsets = np.zeros(len(selected_rows) + 1, dtype=np.int64)
            np.cumsum(lengths, out=self.row_offsets[1:])
            if all_positions is not None:
                self.target_positions = concatenate_or_empty(
                    position_chunks, np.int16
                )
        else:
            self.feature_indices = np.asarray(sequence_index[keep], dtype=np.int64)
            self.labels = np.asarray(labels[keep])
            self.antigen_indices = (
                np.asarray(antigen_index, dtype=np.int64)
                if antigen_index is not None
                else None
            )

        self.vhh_dim = int(self.representations.shape[1])
        self.antigen_dim = int(self.antigens.shape[1]) if self.antigens is not None else 0
        self.input_dim = self.vhh_dim + self.antigen_dim

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index

    def collate(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        feature = np.array(
            self.representations[self.feature_indices[indices]],
            dtype=np.float32,
            copy=True,
        )
        if self.antigen_indices is not None:
            antigen = np.array(
                self.antigens[self.antigen_indices[indices]],
                dtype=np.float32,
                copy=True,
            )
            feature = np.concatenate((feature, antigen), axis=-1)
        if self.kind == "regression":
            target = np.asarray(self.labels[indices], dtype=np.float32)
        else:
            target = np.asarray(self.labels[indices], dtype=np.int64)
        return torch.from_numpy(feature), torch.as_tensor(target)


def concatenate_or_empty(chunks, dtype):
    if not chunks:
        return np.empty(0, dtype=dtype)
    return np.concatenate(chunks).astype(dtype, copy=False)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, values):
        return torch.relu(self.fc2(torch.relu(self.fc1(values))) + values)


class ProbeHead(nn.Module):
    def __init__(self, task, input_dim, vhh_dim, head_style):
        super().__init__()
        self.kind = task["kind"]
        self.vhh_dim = vhh_dim
        self.head_style = head_style
        output_dim = 1 if self.kind in {"regression", "interaction"} else task["classes"]

        self.pooler = None
        self.residual_layers = None
        if head_style == "nbbench" and self.kind == "interaction":
            self.pooler = nn.Linear(vhh_dim, vhh_dim)
        if head_style == "nbbench" and self.kind == "paratope":
            self.residual_layers = nn.Sequential(
                ResidualBlock(input_dim),
                ResidualBlock(input_dim),
                ResidualBlock(input_dim),
            )
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, values):
        if self.pooler is not None:
            vhh = torch.tanh(self.pooler(values[:, : self.vhh_dim]))
            values = torch.cat((vhh, values[:, self.vhh_dim :]), dim=-1)
        if self.residual_layers is not None:
            values = self.residual_layers(values)
        return self.classifier(values)


def focal_binary_loss(logits, targets, gamma=2.0, alpha=0.25):
    targets = targets.float().view_as(logits)
    probabilities = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probability_true = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_true = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_true * (1 - probability_true).pow(gamma) * bce).mean()


def task_loss(task, logits, targets):
    if task["kind"] == "regression":
        return nn.functional.mse_loss(logits.squeeze(-1), targets.float())
    if task["kind"] == "interaction":
        return focal_binary_loss(logits, targets)
    return nn.functional.cross_entropy(logits, targets.long())


def safe_correlation(function, labels, predictions):
    if len(labels) < 2 or np.std(labels) == 0 or np.std(predictions) == 0:
        return float("nan")
    return float(function(labels, predictions).statistic)


def safe_binary_metric(function, labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(function(labels, scores))


def standard_metrics(task, logits, labels):
    kind = task["kind"]
    if kind == "regression":
        predictions = logits.reshape(-1)
        labels = labels.astype(np.float64, copy=False)
        mse = sklearn_metrics.mean_squared_error(labels, predictions)
        pearson = safe_correlation(scipy.stats.pearsonr, labels, predictions)
        return {
            "pearson": pearson,
            "spearman": safe_correlation(scipy.stats.spearmanr, labels, predictions),
            "r2": pearson**2,
            "mae": float(sklearn_metrics.mean_absolute_error(labels, predictions)),
            "mse": float(mse),
            "rmse": float(math.sqrt(mse)),
            "sdr": (
                float(np.std(predictions) / np.std(labels))
                if np.std(labels) != 0
                else float("nan")
            ),
        }

    if kind == "interaction":
        scores = 1 / (1 + np.exp(-logits.reshape(-1)))
        predictions = (scores >= 0.5).astype(np.int64)
    else:
        predictions = np.argmax(logits, axis=-1)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        scores = probabilities[:, 1] if probabilities.shape[1] == 2 else None

    average = (
        "binary"
        if task.get("classes") == 2 and kind != "paratope"
        else "macro"
    )
    result = {
        "accuracy": float(sklearn_metrics.accuracy_score(labels, predictions)),
        "f1": float(
            sklearn_metrics.f1_score(
                labels, predictions, average=average, zero_division=0
            )
        ),
        "matthews_correlation": float(
            sklearn_metrics.matthews_corrcoef(labels, predictions)
        ),
        "precision": float(
            sklearn_metrics.precision_score(
                labels, predictions, average=average, zero_division=0
            )
        ),
        "recall": float(
            sklearn_metrics.recall_score(
                labels, predictions, average=average, zero_division=0
            )
        ),
    }
    if scores is not None:
        result["auprc"] = safe_binary_metric(
            sklearn_metrics.average_precision_score, labels, scores
        )
        result["auroc"] = safe_binary_metric(
            sklearn_metrics.roc_auc_score, labels, scores
        )
    return result


def cdr_metrics(logits, labels, dataset):
    predictions = np.argmax(logits, axis=-1)
    matrix = substitution_matrices.load("BLOSUM62")
    correct = np.zeros(3, dtype=np.int64)
    total = np.zeros(3, dtype=np.int64)
    reconstructions = np.zeros(3, dtype=np.int64)
    blosum_sum = np.zeros(3, dtype=np.float64)
    valid_rows = 0
    for row in range(len(dataset.row_offsets) - 1):
        begin = int(dataset.row_offsets[row])
        end = int(dataset.row_offsets[row + 1])
        positions = dataset.target_positions[begin:end]
        boundaries = np.split(
            np.arange(begin, end), np.flatnonzero(np.diff(positions) != 1) + 1
        )
        if len(boundaries) != 3:
            continue
        valid_rows += 1
        for cdr_index, indices in enumerate(boundaries):
            cdr_labels = labels[indices]
            cdr_predictions = predictions[indices]
            correct[cdr_index] += int((cdr_labels == cdr_predictions).sum())
            total[cdr_index] += len(indices)
            reconstructions[cdr_index] += int(np.array_equal(cdr_labels, cdr_predictions))
            for target, prediction in zip(cdr_labels, cdr_predictions):
                blosum_sum[cdr_index] += matrix[
                    AA_ALPHABET[int(target)], AA_ALPHABET[int(prediction)]
                ]
    denominator_rows = len(dataset.row_offsets) - 1
    return {
        "token_accuracy": float(correct.sum() / total.sum()) if total.sum() else 0.0,
        "cdr1_accuracy": float(correct[0] / total[0]) if total[0] else 0.0,
        "cdr2_accuracy": float(correct[1] / total[1]) if total[1] else 0.0,
        "cdr3_accuracy": float(correct[2] / total[2]) if total[2] else 0.0,
        "cdr1_reconstruction_rate": float(reconstructions[0] / denominator_rows),
        "cdr2_reconstruction_rate": float(reconstructions[1] / denominator_rows),
        "cdr3_reconstruction_rate": float(reconstructions[2] / denominator_rows),
        "overall_cdr_reconstruction_rate": float(
            reconstructions.sum() / (3 * denominator_rows)
        ),
        "blosum_total_mean_score": (
            float(blosum_sum.sum() / total.sum()) if total.sum() else 0.0
        ),
        "cdr1_blosum_mean_score": float(blosum_sum[0] / total[0]) if total[0] else 0.0,
        "cdr2_blosum_mean_score": float(blosum_sum[1] / total[1]) if total[1] else 0.0,
        "cdr3_blosum_mean_score": float(blosum_sum[2] / total[2]) if total[2] else 0.0,
        "valid_three_cdr_rows": valid_rows,
    }


@torch.no_grad()
def evaluate(model, loader, dataset, task, device):
    model.eval()
    logits = []
    labels = []
    loss_sum = 0.0
    count = 0
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(features)
        loss = task_loss(task, outputs, targets)
        batch_size = len(targets)
        loss_sum += float(loss) * batch_size
        count += batch_size
        logits.append(outputs.float().cpu().numpy())
        labels.append(targets.cpu().numpy())
    if not logits:
        raise ValueError(f"empty {dataset.split} dataset after overlap filtering")
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    result = {"loss": loss_sum / count}
    if task["kind"] == "masked_residue_classification":
        result.update(cdr_metrics(logits, labels, dataset))
    else:
        result.update(standard_metrics(task, logits, labels))
    return result


def primary_metric(task, metrics):
    if task["kind"] == "regression":
        return metrics["mse"], False
    if task["kind"] == "masked_residue_classification":
        return metrics["token_accuracy"], True
    return metrics["accuracy"], True


def train_epoch(model, loader, task, optimizer, scheduler, scaler, device, accumulation):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    count = 0
    for step, (features, targets) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            logits = model(features)
            loss = task_loss(task, logits, targets)
            scaled_loss = loss / accumulation
        scaler.scale(scaled_loss).backward()
        if step % accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        loss_sum += float(loss.detach()) * len(targets)
        count += len(targets)
    return loss_sum / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--embedding-dir", default="baselines/nbbench/lmnano_checkpoint18")
    parser.add_argument("--output-dir", default="baselines/nbbench/results")
    parser.add_argument(
        "--overlap-policy",
        choices=("keep", "exclude-test", "exclude-all"),
        default="keep",
    )
    parser.add_argument("--head-style", choices=("linear", "nbbench"), default="nbbench")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="examples per batch; 0 uses 32 sequences, 1024 CDR masks, or 4096 residues",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--feature-storage",
        choices=("memory", "mmap"),
        default="memory",
        help="load shared feature tables into RAM or access them through mmap",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 0 or args.gradient_accumulation <= 0:
        parser.error("batch size must be non-negative and accumulation must be positive")
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    root = os.path.join(args.embedding_dir, args.dataset)
    with open(os.path.join(root, "metadata.json")) as handle:
        export_metadata = json.load(handle)
    task = export_metadata["task"]
    if args.batch_size:
        batch_size = args.batch_size
    elif task["kind"] == "masked_residue_classification":
        batch_size = 1024
    elif task["kind"] in {"residue_classification", "paratope"}:
        batch_size = 4096
    else:
        batch_size = 32
    epochs = args.epochs or (30 if args.dataset == "hIL6" else 50)
    mmap_mode = None if args.feature_storage == "memory" else "r"
    representations = np.load(
        os.path.join(root, "representations.npy"), mmap_mode=mmap_mode
    )
    antigens = None
    if "antigen" in task:
        antigens = np.load(
            os.path.join(root, "antigen_means.npy"), mmap_mode=mmap_mode
        )
    datasets = {
        split: ProbeDataset(
            root,
            split,
            task,
            args.overlap_policy,
            representations,
            antigens,
        )
        for split in ("train", "val", "test")
    }
    train_dataset = datasets["train"]
    for split, dataset in datasets.items():
        if dataset.input_dim != train_dataset.input_dim:
            raise ValueError(f"feature dimensions differ in {split}")
        print(
            f"{split}: examples={len(dataset)}, rows={len(dataset.original_row_indices)}, "
            f"filtered_overlap_rows={dataset.filtered_overlap_rows}",
            flush=True,
        )

    device = torch.device(args.device)
    if args.fp16 and device.type != "cuda":
        parser.error("--fp16 requires a CUDA device")
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=dataset.collate,
        )
        for split, dataset in datasets.items()
    }
    model = ProbeHead(
        task,
        train_dataset.input_dim,
        train_dataset.vhh_dim,
        args.head_style,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(
        len(loaders["train"]) / args.gradient_accumulation
    )
    total_updates = max(1, updates_per_epoch * epochs)

    def schedule(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return max(
            0.0,
            (total_updates - step) / max(1, total_updates - args.warmup_steps),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    output_dir = os.path.join(
        args.output_dir,
        args.dataset,
        f"{args.head_style}_{args.overlap_policy}_seed{args.seed}",
    )
    if os.path.isdir(output_dir) and os.listdir(output_dir) and not args.overwrite:
        raise FileExistsError(f"non-empty output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    best_state = None
    best_epoch = None
    best_value = None
    best_validation = None
    stale_epochs = 0
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            loaders["train"],
            task,
            optimizer,
            scheduler,
            scaler,
            device,
            args.gradient_accumulation,
        )
        validation = evaluate(
            model, loaders["val"], datasets["val"], task, device
        )
        value, maximize = primary_metric(task, validation)
        improved = best_value is None or (value > best_value if maximize else value < best_value)
        if improved:
            best_value = value
            best_epoch = epoch
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation,
            "best_epoch": best_epoch,
        }
        history.append(record)
        print(json.dumps(record, allow_nan=True), flush=True)
        if stale_epochs >= args.early_stopping_patience:
            print(f"early stopping after epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model, loaders["test"], datasets["test"], task, device
    )
    checkpoint_path = os.path.join(output_dir, "best_head.pt")
    torch.save(
        {
            "model": best_state,
            "task": task,
            "input_dim": train_dataset.input_dim,
            "vhh_dim": train_dataset.vhh_dim,
            "head_style": args.head_style,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    result = {
        "dataset": args.dataset,
        "task": task,
        "overlap_policy": args.overlap_policy,
        "head_style": args.head_style,
        "seed": args.seed,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "test": test_metrics,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split_examples": {split: len(dataset) for split, dataset in datasets.items()},
        "split_rows": {
            split: len(dataset.original_row_indices) for split, dataset in datasets.items()
        },
        "filtered_overlap_rows": {
            split: dataset.filtered_overlap_rows for split, dataset in datasets.items()
        },
        "export_metadata": export_metadata,
        "arguments": vars(args),
        "resolved_batch_size": batch_size,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, indent=2))
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
