#!/usr/bin/env python3
"""Train DiffAb fix-backbone CDR-H3 from scratch on the clustered LmPro split.

The official ``fixbb.pt`` file supplies only the published architecture/configuration;
its model parameters are deliberately not loaded because its SAbDab snapshot overlaps
the clean test set.  Inputs are built by ``run_diffab_benchmark.py`` from the same
record-level atom14 data and 20-residue epitope context as LmPro-Nano.
"""

import argparse
import copy
import contextlib
import json
import math
import os
import random
import sys
import time

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
from diffab.utils.train import recursive_to  # noqa: E402
from diffab.utils.transforms import MaskSingleCDR, MergeChains, PatchAroundAnchor  # noqa: E402
from record_ids import load_record_subset  # noqa: E402
from run_diffab_benchmark import build_structure, seed_all  # noqa: E402


def prepare_training_data(structure):
    variant = MaskSingleCDR("H_CDR3", augmentation=False)(copy.deepcopy(structure))
    merged = MergeChains()(variant)
    return PatchAroundAnchor()(merged)


def load_dataset(source_jsonl, record_path, epitope_size, limit=None):
    selected = load_record_subset(record_path)
    examples = []
    with open(source_jsonl) as handle:
        for source_line, line in enumerate(handle):
            if source_line not in selected:
                continue
            if limit is not None and len(examples) >= limit:
                break
            structure, metadata = build_structure(
                json.loads(line), source_line, epitope_size
            )
            examples.append((prepare_training_data(structure), metadata))
    expected = min(len(selected), limit) if limit is not None else len(selected)
    if len(examples) != expected:
        raise ValueError(f"expected {expected} examples from {source_jsonl}, got {len(examples)}")
    return examples


def iter_batches(examples, batch_size, order):
    collate = PaddingCollate(eight=False)
    for start in range(0, len(order), batch_size):
        yield collate([examples[i][0] for i in order[start : start + batch_size]])


def weighted_loss(losses, weights):
    total = None
    for name, weight in weights.items():
        value = losses[name] * float(weight)
        total = value if total is None else total + value
    return total


def evaluate(model, examples, batch_size, device, weights, seed):
    seed_all(seed)
    model.eval()
    loss_sum = positions = 0.0
    component_sums = {name: 0.0 for name in weights}
    with torch.no_grad():
        order = list(range(len(examples)))
        for batch in iter_batches(examples, batch_size, order):
            batch = recursive_to(batch, device)
            count = float(batch["generate_flag"].sum().cpu())
            losses = model(batch)
            total = weighted_loss(losses, weights)
            loss_sum += float(total.cpu()) * count
            for name in component_sums:
                component_sums[name] += float(losses[name].cpu()) * count
            positions += count
    return loss_sum / positions, {name: value / positions for name, value in component_sums.items()}, int(positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--valid-jsonl", required=True)
    parser.add_argument("--valid-records", required=True)
    parser.add_argument("--architecture-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epitope-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=100.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    args = parser.parse_args()

    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        parser.error("epochs, patience, and batch size must be positive")
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)

    # Model construction initializes every DiffAb parameter, so seed before it.
    seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    architecture = torch.load(args.architecture_checkpoint, map_location="cpu")
    config = architecture["config"]
    model = get_model(config.model).to(device)  # random initialization: do not load state_dict
    weights = dict(config.train.loss_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(float(config.train.optimizer.beta1), float(config.train.optimizer.beta2)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.train.scheduler.factor),
        patience=int(config.train.scheduler.patience),
        min_lr=float(config.train.scheduler.min_lr),
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("building record-aligned DiffAb datasets", flush=True)
    train_examples = load_dataset(
        args.train_jsonl, args.train_records, args.epitope_size, args.train_limit
    )
    valid_examples = load_dataset(
        args.valid_jsonl, args.valid_records, args.epitope_size, args.valid_limit
    )
    print(
        f"train={len(train_examples)} valid={len(valid_examples)} device={device} seed={args.seed}",
        flush=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training.jsonl")
    best_path = os.path.join(args.output_dir, "best.pt")
    best_loss = math.inf
    stale = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        seed_all(args.seed + epoch)
        order = list(range(len(train_examples)))
        random.shuffle(order)
        model.train()
        train_sum = train_positions = 0.0
        for batch in iter_batches(train_examples, args.batch_size, order):
            batch = recursive_to(batch, device)
            positions = float(batch["generate_flag"].sum().cpu())
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()
            with amp_context:
                losses = model(batch)
                loss = weighted_loss(losses, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.detach().cpu()) * positions
            train_positions += positions
            global_step += 1

        valid_loss, valid_components, valid_positions = evaluate(
            model,
            valid_examples,
            args.batch_size,
            device,
            weights,
            args.seed + 1_000_000,
        )
        scheduler.step(valid_loss)
        row = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_sum / train_positions,
            "valid_loss": valid_loss,
            "valid_components": valid_components,
            "train_positions": int(train_positions),
            "valid_positions": valid_positions,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        with open(log_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        if valid_loss < best_loss:
            best_loss = valid_loss
            stale = 0
            torch.save(
                {
                    "config": config,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "iteration": global_step,
                    "avg_val_loss": best_loss,
                    "training_provenance": {
                        "initialization": "random",
                        "architecture_only_from": os.path.abspath(args.architecture_checkpoint),
                        "train_jsonl": os.path.abspath(args.train_jsonl),
                        "train_records": os.path.abspath(args.train_records),
                        "valid_jsonl": os.path.abspath(args.valid_jsonl),
                        "valid_records": os.path.abspath(args.valid_records),
                        "seed": args.seed,
                    },
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}; best valid loss={best_loss:.6f}")
                break
    print(f"best checkpoint: {best_path} (valid loss={best_loss:.6f})")


if __name__ == "__main__":
    main()
