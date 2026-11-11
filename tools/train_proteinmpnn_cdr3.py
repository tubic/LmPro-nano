#!/usr/bin/env python3
"""Train a leakage-clean ProteinMPNN baseline on record-aligned CDR-H3 targets.

Unlike the official checkpoint, this model starts from random weights and sees only the
specified clustered training export.  The architecture and smoothed NLL are ProteinMPNN's;
the loss mask is restricted to CDR-H3 so that the objective matches LmPro-Nano.
"""

import argparse
import copy
import contextlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTEINMPNN = os.path.join(ROOT, "third_party", "ProteinMPNN")
sys.path.insert(0, PROTEINMPNN)

from protein_mpnn_utils import (  # noqa: E402
    ProteinMPNN,
    StructureDatasetPDB,
    loss_smoothed,
    tied_featurize,
)


def load_export(directory, max_records=None):
    with open(os.path.join(directory, "parsed_pdbs.jsonl")) as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    with open(os.path.join(directory, "assigned_chains.jsonl")) as handle:
        chain_dict = json.load(handle)
    with open(os.path.join(directory, "fixed_positions.jsonl")) as handle:
        fixed_dict = json.load(handle)
    if max_records is not None:
        entries = entries[:max_records]
    dataset = StructureDatasetPDB(
        entries, truncate=None, max_length=10000, verbose=False
    )
    if len(dataset) != len(entries):
        raise ValueError(f"ProteinMPNN rejected {len(entries) - len(dataset)} records from {directory}")
    names = {entry["name"] for entry in entries}
    if not names.issubset(chain_dict) or not names.issubset(fixed_dict):
        raise ValueError(f"structure/chain/fixed-position keys disagree in {directory}")
    chain_dict = {name: chain_dict[name] for name in names}
    fixed_dict = {name: fixed_dict[name] for name in names}
    return dataset, chain_dict, fixed_dict


def batch_signature(entry, chain_dict):
    """tied_featurize derives one shared chain layout from the last batch entry,
    so a batch may only mix records with identical masked/visible chain letters."""
    masked, visible = chain_dict[entry["name"]]
    seq_keys = tuple(sorted(k[10:] for k in entry if k.startswith("seq_chain_")))
    return (tuple(masked), tuple(visible), seq_keys)


def batches(dataset, batch_size, order, chain_dict):
    buckets = {}
    for i in order:
        buckets.setdefault(batch_signature(dataset[i], chain_dict), []).append(i)
    for sig in sorted(buckets):
        group = buckets[sig]
        for start in range(0, len(group), batch_size):
            yield [copy.deepcopy(dataset[i]) for i in group[start : start + batch_size]]


def featurize(batch, device, chain_dict, fixed_dict):
    return tied_featurize(
        batch,
        device,
        chain_dict,
        fixed_position_dict=fixed_dict,
        omit_AA_dict=None,
        tied_positions_dict=None,
        pssm_dict=None,
        bias_by_res_dict=None,
        ca_only=False,
    )


def unpack(features):
    if len(features) != 20:
        raise ValueError(f"expected 20 ProteinMPNN features, got {len(features)}")
    x = features[0]
    sequence = features[1]
    mask = features[2]
    chain_mask = features[4]
    chain_encoding = features[5]
    position_mask = features[10]
    residue_index = features[12]
    design_mask = mask * chain_mask * position_mask
    return x, sequence, mask, design_mask, residue_index, chain_encoding


def validation(model, dataset, batch_size, device, chain_dict, fixed_dict):
    model.eval()
    nll_sum = matches = positions = 0.0
    with torch.no_grad():
        for batch in batches(dataset, batch_size, list(range(len(dataset))), chain_dict):
            x, sequence, mask, design_mask, residue_index, chain_encoding = unpack(
                featurize(batch, device, chain_dict, fixed_dict)
            )
            # A fixed non-random decoding order makes checkpoint selection repeatable.
            order_noise = torch.arange(sequence.size(1), device=device, dtype=torch.float32)
            order_noise = order_noise.unsqueeze(0).expand_as(design_mask)
            log_probs = model(
                x,
                sequence,
                mask,
                design_mask,
                residue_index,
                chain_encoding,
                order_noise,
            )
            token_nll = F.nll_loss(
                log_probs.reshape(-1, log_probs.size(-1)),
                sequence.reshape(-1),
                reduction="none",
            ).view_as(sequence)
            nll_sum += float((token_nll * design_mask).sum().cpu())
            prediction = log_probs.argmax(dim=-1)
            matches += float(((prediction == sequence) * design_mask.bool()).sum().cpu())
            positions += float(design_mask.sum().cpu())
    return nll_sum / positions, matches / positions, int(positions)


def noam_rate(step, model_size, factor, warmup):
    """ProteinMPNN's official inverse-square-root schedule."""
    step = max(1, step)
    return factor * model_size ** (-0.5) * min(
        step ** (-0.5), step * warmup ** (-1.5)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--valid-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=("noam", "adam"), default="noam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--lr-factor", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--backbone-noise", type=float, default=0.2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-valid-records", type=int, default=None)
    args = parser.parse_args()

    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        parser.error("epochs, patience, and batch size must be positive")
    if args.warmup_steps <= 0 or args.lr_factor <= 0:
        parser.error("warmup steps and LR factor must be positive")
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    for value, name in (
        (args.max_train_records, "--max-train-records"),
        (args.max_valid_records, "--max-valid-records"),
    ):
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)

    train_set, train_chains, train_fixed = load_export(
        args.train_input, args.max_train_records
    )
    valid_set, valid_chains, valid_fixed = load_export(
        args.valid_input, args.max_valid_records
    )
    model = ProteinMPNN(
        num_letters=21,
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        k_neighbors=args.neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
    ).to(device)
    if args.optimizer == "noam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training.jsonl")
    best_path = os.path.join(args.output_dir, "best.pt")
    best_loss = float("inf")
    stale = 0
    global_step = 0

    print(f"train={len(train_set)} valid={len(valid_set)} device={device} seed={args.seed}")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        order = list(range(len(train_set)))
        random.shuffle(order)
        train_nll_sum = train_positions = 0.0
        for batch in batches(train_set, args.batch_size, order, train_chains):
            x, sequence, mask, design_mask, residue_index, chain_encoding = unpack(
                featurize(batch, device, train_chains, train_fixed)
            )
            order_noise = torch.randn_like(design_mask)
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()
            )
            with amp_context:
                log_probs = model(
                    x,
                    sequence,
                    mask,
                    design_mask,
                    residue_index,
                    chain_encoding,
                    order_noise,
                )
                token_loss, smoothed_loss = loss_smoothed(
                    sequence, log_probs, design_mask
                )
            scaler.scale(smoothed_loss).backward()
            scaler.unscale_(optimizer)
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            global_step += 1
            if args.optimizer == "noam":
                learning_rate = noam_rate(
                    global_step,
                    args.hidden_dim,
                    args.lr_factor,
                    args.warmup_steps,
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
            scaler.step(optimizer)
            scaler.update()
            train_nll_sum += float((token_loss * design_mask).sum().detach().cpu())
            train_positions += float(design_mask.sum().cpu())

        valid_loss, valid_aar, valid_positions = validation(
            model, valid_set, args.batch_size, device, valid_chains, valid_fixed
        )
        row = {
            "epoch": epoch,
            "train_nll": train_nll_sum / train_positions,
            "valid_nll": valid_loss,
            "valid_teacher_forced_aar": valid_aar,
            "lr": optimizer.param_groups[0]["lr"],
            "step": global_step,
            "train_positions": int(train_positions),
            "valid_positions": valid_positions,
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
                    "epoch": epoch,
                    "step": global_step,
                    "num_edges": args.neighbors,
                    "noise_level": args.backbone_noise,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_valid_nll": best_loss,
                    "training_provenance": {
                        "initialization": "random",
                        "train_input": os.path.abspath(args.train_input),
                        "valid_input": os.path.abspath(args.valid_input),
                        "seed": args.seed,
                        "optimizer": args.optimizer,
                        "warmup_steps": args.warmup_steps,
                        "lr_factor": args.lr_factor,
                        "max_train_records": args.max_train_records,
                        "max_valid_records": args.max_valid_records,
                    },
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}; best valid NLL={best_loss:.6f}")
                break
    print(f"best checkpoint: {best_path} (valid NLL={best_loss:.6f})")


if __name__ == "__main__":
    main()
