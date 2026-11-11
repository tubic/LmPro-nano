#!/usr/bin/env python3
"""Reviewer-grade reproduction of the single-mutation Table 3 protocol.

The default run reproduces the published RDE-PPI three-fold complex split.
Each fold model is trained for a fixed, predeclared number of updates, and no
outer-fold prediction is computed until all model weights are locked.
"""

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from rde.datasets.skempi import SkempiDataset
from rde.utils.data import PaddingCollate
from rde.utils.misc import get_logger, get_new_log_dir, seed_all
from rde.utils.skempi import eval_skempi_three_modes
from rde.utils.skempi_splits import (
    audit_folds,
    balanced_pdb_folds,
    rde_complex_folds,
)
from rde.utils.train import recursive_to
from rde.utils.transforms import get_transform


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/train/profe_rde_table3.yml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rde-checkpoint", default=None)
    parser.add_argument("--logdir", default="outputs/table3_runs")
    parser.add_argument("--tag", default="")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps-per-fold", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=1000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--protocol", choices=["rde3", "pdb5"], default=None)
    parser.add_argument(
        "--no-context-precompute",
        action="store_true",
        help="Compute the frozen backbone live; intended only for smoke tests.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Smoke-test only: truncate each outer fold evaluation.",
    )
    return parser.parse_args()


def resolve_model_factory(name):
    if name == "rde_profe":
        from rde.models.rde_profe import model_factory

        return model_factory
    if name == "profe_enhanced":
        from rde.models.rde_2025 import model_factory

        return model_factory
    raise ValueError("Unsupported model.factory: %s" % name)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def worker_seed(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def infinite_batches(loader):
    while True:
        yield from loader


class ContextCacheDataset(Dataset):
    def __init__(self, dataset, context_cache):
        self.dataset = dataset
        self.context_cache = context_cache

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        sample_id = int(item["id"])
        if sample_id not in self.context_cache:
            raise KeyError("Missing backbone context for sample id %d" % sample_id)
        wt_context, mt_context = self.context_cache[sample_id]
        item["x_pret_wt"] = wt_context
        item["x_pret_mt"] = mt_context
        return item


@torch.no_grad()
def precompute_backbone_contexts(
    model, source_datasets, device, batch_size, num_workers, logger
):
    if not hasattr(model, "_get_rde_context"):
        raise TypeError("Selected model does not expose _get_rde_context")
    if batch_size < 1:
        raise ValueError("training.context_batch_size must be positive")

    model.eval()
    cache = {}
    processed = 0
    for source_index, dataset in enumerate(source_datasets):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=PaddingCollate(),
            num_workers=num_workers,
            worker_init_fn=worker_seed,
            pin_memory=device.type == "cuda",
        )
        for batch in loader:
            sample_ids = [int(value) for value in batch["id"].tolist()]
            lengths = [int(value) for value in batch["mask"].sum(dim=1).tolist()]
            batch = recursive_to(batch, device)
            aa_wt = batch["aa_wt"] if "aa_wt" in batch else batch["aa"]
            wt_context = model._get_rde_context({**batch, "aa": aa_wt})
            mt_context = model._get_rde_context({**batch, "aa": batch["aa_mut"]})
            if not torch.isfinite(wt_context).all() or not torch.isfinite(mt_context).all():
                raise FloatingPointError("Non-finite frozen-backbone context")
            for row, (sample_id, length) in enumerate(zip(sample_ids, lengths)):
                cache[sample_id] = (
                    wt_context[row, :length].detach().to(dtype=torch.float16, device="cpu").contiguous(),
                    mt_context[row, :length].detach().to(dtype=torch.float16, device="cpu").contiguous(),
                )
            processed += len(sample_ids)
            if processed == len(sample_ids) or processed % 400 < len(sample_ids):
                logger.info(
                    "Backbone context cache: source=%d processed=%d cached=%d",
                    source_index,
                    processed,
                    len(cache),
                )

    expected = sum(len(dataset) for dataset in source_datasets)
    if len(cache) != expected:
        raise RuntimeError(
            "Expected %d unique cached contexts, found %d" % (expected, len(cache))
        )
    logger.info("Frozen-backbone context cache complete: %d samples", len(cache))
    return cache


def serialize_config(config):
    if isinstance(config, dict):
        return {key: serialize_config(value) for key, value in config.items()}
    if isinstance(config, list):
        return [serialize_config(value) for value in config]
    return config


def train_accumulated_update(
    model,
    optimizer,
    iterator,
    device,
    accumulation_steps,
    max_grad_norm,
    iteration,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0

    # Pearson/ranking losses are nonlinear in the batch.  For opt-in models,
    # retain the branch predictions across microbatches and evaluate the loss
    # once on the full effective batch.  The historical path below remains
    # unchanged for locked legacy configurations.
    aggregate_loss = bool(getattr(model, "aggregate_loss_across_accumulation", False))
    if aggregate_loss and not hasattr(model, "loss_from_branches"):
        raise TypeError(
            "aggregate_loss_across_accumulation requires loss_from_branches"
        )
    fwd_parts, rev_parts, target_parts = [], [], []
    for micro_step in range(accumulation_steps):
        batch = recursive_to(next(iterator), device)
        losses, output = model(batch)
        if aggregate_loss:
            required = ("_loss_fwd", "_loss_rev", "_loss_target")
            if any(key not in output for key in required):
                raise RuntimeError(
                    "Aggregated-loss model did not return branch predictions"
                )
            fwd_parts.append(output["_loss_fwd"])
            rev_parts.append(output["_loss_rev"])
            target_parts.append(output["_loss_target"])
        else:
            loss = losses["total_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss at iteration %d micro-step %d"
                    % (iteration, micro_step)
                )
            (loss / accumulation_steps).backward()
            accumulated_loss += float(loss.detach())

    if aggregate_loss:
        loss = model.loss_from_branches(
            torch.cat(fwd_parts, dim=0),
            torch.cat(rev_parts, dim=0),
            torch.cat(target_parts, dim=0),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite aggregated loss at iteration %d" % iteration)
        loss.backward()
        accumulated_loss = float(loss.detach())

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    grad_norm = clip_grad_norm_(trainable, max_grad_norm)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("Non-finite gradient at iteration %d" % iteration)
    optimizer.step()
    reported_loss = accumulated_loss if aggregate_loss else accumulated_loss / accumulation_steps
    return reported_loss, grad_norm


def save_checkpoint(path, config, args, models, optimizers, iteration, generators):
    checkpoint = {
        "schema_version": 1,
        "config": config,
        "cli_args": vars(args),
        "iteration": iteration,
        "next_iteration": iteration + 1,
        "models": [model.state_dict() for model in models],
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "loader_rng": [generator.get_state() for generator in generators],
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path)


def restore_checkpoint(path, models, optimizers, generators):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for model, state in zip(models, checkpoint["models"]):
        model.load_state_dict(state, strict=True)
    for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
        optimizer.load_state_dict(state)
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    for generator, state in zip(generators, checkpoint["loader_rng"]):
        generator.set_state(state)
    cuda_rng = checkpoint.get("cuda_rng")
    if cuda_rng is not None and len(cuda_rng) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda_rng)
    return int(checkpoint.get("next_iteration", checkpoint["iteration"] + 1))


def main():
    args = parse_args()
    with open(args.config) as handle:
        raw = yaml.safe_load(handle)
    if args.rde_checkpoint is not None:
        raw["model"]["rde_checkpoint"] = args.rde_checkpoint
    if args.steps_per_fold is not None:
        raw["training"]["steps_per_fold"] = args.steps_per_fold
    if args.batch_size is not None:
        raw["training"]["batch_size"] = args.batch_size
    if args.accumulation_steps is not None:
        raw["training"]["accumulation_steps"] = args.accumulation_steps
    if args.learning_rate is not None:
        raw["training"]["learning_rate"] = args.learning_rate
    if args.seed is not None:
        raw["training"]["seed"] = args.seed
    if args.protocol is not None:
        raw["dataset"]["protocol"] = args.protocol
    factory_name = raw["model"].get("factory")
    model_factory = resolve_model_factory(factory_name)
    accumulation_steps = int(raw["training"].get("accumulation_steps", 1))
    if (
        args.num_workers < 0
        or raw["training"]["steps_per_fold"] < 1
        or accumulation_steps < 1
    ):
        raise ValueError("Worker and step counts are invalid")
    if args.checkpoint_freq < 1:
        raise ValueError("--checkpoint-freq must be positive")
    if args.max_val_batches is not None and args.max_val_batches < 1:
        raise ValueError("--max-val-batches must be positive")

    from easydict import EasyDict

    config = EasyDict(raw)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_all(config.training.seed)

    protocol = str(getattr(config.dataset, "protocol", "rde3"))
    if protocol not in {"rde3", "pdb5"}:
        raise ValueError("dataset.protocol must be rde3 or pdb5")
    num_folds = 3 if protocol == "rde3" else 5
    group_field = "complex" if protocol == "rde3" else "pdbcode"
    protocol_description = (
        "RDE-PPI three-fold complex shuffle"
        if protocol == "rde3"
        else "balanced five-fold PDB-disjoint split"
    )

    run_dir = get_new_log_dir(
        args.logdir,
        prefix="table3-%s-%s" % (factory_name, protocol),
        tag=args.tag,
    )
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = get_logger("table3", run_dir)
    logger.info("Run directory: %s", run_dir)
    logger.info("Effective config: %s", config)

    transform = get_transform(
        [
            {"type": "select_atom", "resolution": "backbone+CB"},
            {
                "type": "selected_region_fixed_size_patch",
                "select_attr": "mut_flag",
                "patch_size": 128,
            },
        ]
    )
    reference_dataset = SkempiDataset(
        csv_path=config.dataset.csv_path,
        pdb_dir=config.dataset.pdb_dir,
        cache_dir=config.dataset.cache_dir,
        num_cvfolds=3,
        cvfold_index=0,
        split="train",
        split_seed=config.dataset.split_seed,
        transform=None,
    )
    entries = reference_dataset.entries_full
    outer_groups = (
        rde_complex_folds(entries, num_folds, config.dataset.split_seed)
        if protocol == "rde3"
        else balanced_pdb_folds(entries, num_folds, config.dataset.split_seed)
    )
    split_audit = audit_folds(entries, outer_groups, group_field)

    dataset_options = {
        "csv_path": config.dataset.csv_path,
        "pdb_dir": config.dataset.pdb_dir,
        "cache_dir": config.dataset.cache_dir,
        "num_cvfolds": num_folds,
        "split_seed": config.dataset.split_seed,
        "transform": transform,
        "fold_groups": outer_groups,
        "fold_group_field": group_field,
    }
    logger.info("Loading SKEMPI datasets for protocol %s...", protocol)
    train_datasets = [
        SkempiDataset(cvfold_index=fold, split="train", **dataset_options)
        for fold in range(num_folds)
    ]
    val_datasets = [
        SkempiDataset(cvfold_index=fold, split="val", **dataset_options)
        for fold in range(num_folds)
    ]

    duplicate_site_rows = []
    for entry in entries:
        sites = {
            (mutation["chain"], mutation["resseq"], mutation["icode"])
            for mutation in entry["mutations"]
        }
        if len(sites) != len(entry["mutations"]):
            duplicate_site_rows.append(
                {
                    "id": int(entry["id"]),
                    "complex": entry["complex"],
                    "mutstr": entry["mutstr"],
                    "listed_mutations": len(entry["mutations"]),
                    "unique_sites": len(sites),
                }
            )
    split_payload = {
        "schema_version": 1,
        "protocol": protocol_description,
        "seed": config.dataset.split_seed,
        "audit": split_audit,
        "annotation_audit": {
            "duplicate_site_rows": duplicate_site_rows,
            "policy": "same-site same-target duplicates are applied once; conflicts fail",
        },
        "folds": [],
    }
    for fold, groups in enumerate(outer_groups):
        fold_entries = [
            entry for entry in entries if str(entry[group_field]) in groups
        ]
        split_payload["folds"].append(
            {
                "fold": fold,
                "complexes": sorted(groups),
                "pdbs": sorted({entry["pdbcode"] for entry in fold_entries}),
                "sample_ids": sorted(int(entry["id"]) for entry in fold_entries),
                "single_mutations": sum(entry["num_muts"] == 1 for entry in fold_entries),
                "multiple_mutations": sum(entry["num_muts"] > 1 for entry in fold_entries),
            }
        )
    with open(os.path.join(run_dir, "split_manifest.json"), "w") as handle:
        json.dump(split_payload, handle, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as handle:
        json.dump(
            {
                "schema_version": 1,
                "config_path": str(Path(args.config).resolve()),
                "config_sha256": sha256_file(args.config),
                "skempi_csv_sha256": sha256_file(config.dataset.csv_path),
                "rde_checkpoint_sha256": sha256_file(config.model.rde_checkpoint),
                "model_factory": factory_name,
                "effective_config": serialize_config(raw),
                "cli_args": vars(args),
                "outer_evaluation_policy": "fixed steps; outer folds opened once after lock",
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    logger.info("Split audit: %s", split_audit)

    models = [model_factory(config).to(device) for _ in range(num_folds)]
    should_precompute = bool(
        getattr(config.training, "precompute_backbone_context", False)
    ) and not args.no_context_precompute
    if should_precompute:
        context_batch_size = int(
            getattr(config.training, "context_batch_size", config.training.batch_size)
        )
        context_cache = precompute_backbone_contexts(
            models[0],
            [train_datasets[0], val_datasets[0]],
            device,
            context_batch_size,
            args.num_workers,
            logger,
        )
        train_datasets = [
            ContextCacheDataset(dataset, context_cache) for dataset in train_datasets
        ]
        val_datasets = [
            ContextCacheDataset(dataset, context_cache) for dataset in val_datasets
        ]

    generators = []
    train_loaders = []
    val_loaders = []
    for fold in range(num_folds):
        generator = torch.Generator().manual_seed(config.training.seed + fold)
        generators.append(generator)
        common = {
            "batch_size": config.training.batch_size,
            "collate_fn": PaddingCollate(),
            "num_workers": args.num_workers,
            "worker_init_fn": worker_seed,
            "pin_memory": device.type == "cuda",
            "generator": generator,
        }
        train_loaders.append(
            DataLoader(train_datasets[fold], shuffle=True, **common)
        )
        val_loaders.append(DataLoader(val_datasets[fold], shuffle=False, **common))

    optimizers = []
    optimizer_name = str(getattr(config.training, "optimizer", "adam")).lower()
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError("training.optimizer must be adam or adamw")
    for model in models:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizers.append(
            optimizer_class(
                trainable,
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        )
    iterators = [infinite_batches(loader) for loader in train_loaders]
    first_iteration = 1
    if args.resume:
        first_iteration = restore_checkpoint(
            args.resume, models, optimizers, generators
        )
        logger.info("Resuming at global iteration %d", first_iteration)

    total_iterations = int(config.training.steps_per_fold) * num_folds
    for iteration in range(first_iteration, total_iterations + 1):
        fold = (iteration - 1) % num_folds
        model, optimizer = models[fold], optimizers[fold]
        accumulated_loss, grad_norm = train_accumulated_update(
            model,
            optimizer,
            iterators[fold],
            device,
            accumulation_steps,
            config.training.max_grad_norm,
            iteration,
        )
        if iteration == 1 or iteration % 100 == 0:
            logger.info(
                "Iter %d/%d fold=%d loss=%.5f grad=%.4f",
                iteration,
                total_iterations,
                fold,
                accumulated_loss,
                float(grad_norm),
            )
        if iteration % args.checkpoint_freq == 0 or iteration == total_iterations:
            path = os.path.join(checkpoint_dir, "iter_%08d.pt" % iteration)
            save_checkpoint(
                path, config, args, models, optimizers, iteration, generators
            )
            save_checkpoint(
                os.path.join(checkpoint_dir, "last.pt"),
                config,
                args,
                models,
                optimizers,
                iteration,
                generators,
            )
            logger.info("Saved checkpoint: %s", path)

    selection_lock = {
        "schema_version": 1,
        "selection_scope": "fixed-predeclared-steps",
        "protocol": protocol,
        "num_folds": num_folds,
        "steps_per_fold": int(config.training.steps_per_fold),
        "physical_batch_size": int(config.training.batch_size),
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": int(config.training.batch_size) * accumulation_steps,
        "outer_predictions_opened": False,
        "checkpoint": os.path.join(checkpoint_dir, "last.pt"),
    }
    with open(os.path.join(run_dir, "selection.lock.json"), "w") as handle:
        json.dump(selection_lock, handle, indent=2, sort_keys=True)

    rows = []
    for fold, (model, loader) in enumerate(zip(models, val_loaders)):
        model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(
                tqdm(loader, desc="Outer fold %d" % fold, dynamic_ncols=True)
            ):
                batch = recursive_to(batch, device)
                _, output = model(batch)
                predictions = output["ddg"].detach().cpu().numpy()
                targets = batch["ddG"].detach().cpu().numpy()
                sample_ids = batch["id"].detach().cpu().numpy()
                for sample_id, complex_id, mutstr, target, prediction in zip(
                    sample_ids,
                    batch["complex"],
                    batch["mutstr"],
                    targets,
                    predictions,
                ):
                    rows.append(
                        {
                            "sample_id": int(sample_id),
                            "fold": fold,
                            "method": str(
                                getattr(config.model, "method_name", factory_name)
                            ),
                            "complex": complex_id,
                            "mutstr": mutstr,
                            "num_muts": len(mutstr.split(",")),
                            "ddG": float(target),
                            "ddG_pred": float(prediction),
                        }
                    )
                if (
                    args.max_val_batches is not None
                    and batch_index + 1 >= args.max_val_batches
                ):
                    break

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(run_dir, "outer_predictions.csv"), index=False)
    if args.max_val_batches is not None:
        logger.info(
            "Smoke evaluation wrote %d predictions; protocol metrics are intentionally "
            "skipped for truncated outer folds.",
            len(results),
        )
        print("RUN_DIR=%s" % run_dir)
        return
    pooled_metrics = eval_skempi_three_modes(results)
    pooled_metrics.to_csv(os.path.join(run_dir, "table3_metrics.csv"), index=False)
    per_fold_metrics = []
    for fold in range(num_folds):
        metrics = eval_skempi_three_modes(results.query("fold == @fold"))
        metrics["fold"] = fold
        per_fold_metrics.append(metrics)
    per_fold_metrics = pd.concat(per_fold_metrics, ignore_index=True)
    per_fold_metrics.to_csv(
        os.path.join(run_dir, "table3_metrics_per_fold.csv"), index=False
    )
    logger.info("Pooled metrics:\n%s", pooled_metrics.to_string(index=False))
    print("RUN_DIR=%s" % run_dir)


if __name__ == "__main__":
    main()
