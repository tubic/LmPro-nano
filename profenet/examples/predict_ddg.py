#!/usr/bin/env python3
"""Predict mutation effects (ddG) with the released ProFE-Net weights.

Runs the exact locked evaluation path (same featurization, transforms, and
fold models as the released Table 3 predictions) on a user-supplied complex.

Usage:

    python profenet/examples/predict_ddg.py \
        --pdb /path/to/complex.pdb \
        --ligand-chains HL --receptor-chains A \
        --mutations LA56G,LA56S

Outputs the per-fold and mean predicted ddG in kcal/mol, using the SKEMPI
convention ddG = dG_mutant - dG_wildtype (positive = predicted weaker
binding, negative = predicted stronger binding). Averaging the three fold
models is the recommended point estimate for new complexes.

Run from the repository root after installing the environment documented in
the root README and placing the released weights under `profenet/weights/`.
"""

import argparse
import os
import sys
import tempfile

import torch
import yaml
from easydict import EasyDict

PROFENET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROFENET_ROOT)

from rde.datasets.skempi import SkempiDataset
from rde.models.rde_2025 import model_factory
from rde.utils.data import PaddingCollate
from rde.utils.train import recursive_to
from rde.utils.transforms import get_transform
from torch.utils.data import DataLoader


def build_entry_csv(workdir, pdb_path, pdbcode, ligand_chains, receptor_chains, mutations):
    """Write a one-row SKEMPI-style CSV and place the structure where the
    dataset parser expects it. Affinities are dummies; they set the (unused)
    target to zero and must be finite and positive."""
    pdb_dir = os.path.join(workdir, "pdbs")
    os.makedirs(pdb_dir, exist_ok=True)
    target_pdb = os.path.join(pdb_dir, f"{pdbcode.upper()}.pdb")
    with open(pdb_path) as src, open(target_pdb, "w") as dst:
        dst.write(src.read())

    complex_id = f"{pdbcode}_{ligand_chains}_{receptor_chains}"
    csv_path = os.path.join(workdir, "entries.csv")
    with open(csv_path, "w") as handle:
        handle.write("#Pdb;Mutation(s)_cleaned;Affinity_wt_parsed;Affinity_mut_parsed\n")
        handle.write(f"{complex_id};{mutations};1e-9;1e-9\n")
    return csv_path, pdb_dir, complex_id


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", default=os.path.join(PROFENET_ROOT, "weights"))
    parser.add_argument(
        "--config",
        default=os.path.join(
            PROFENET_ROOT, "configs", "train", "profe_enhanced_table3.yml"
        ),
    )
    parser.add_argument("--pdb", required=True, help="complex structure (PDB format)")
    parser.add_argument("--ligand-chains", required=True,
                        help="chain letters of the mutated partner, e.g. HL")
    parser.add_argument("--receptor-chains", required=True,
                        help="chain letters of the other partner, e.g. A")
    parser.add_argument("--mutations", required=True,
                        help="comma-separated mutations, e.g. LA56G or LA56G,TS57A "
                             "(WT letter, chain, residue number, mutant letter)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default=None,
                        help="where to build the feature cache (default: temp dir)")
    args = parser.parse_args()

    args.weights_dir = os.path.abspath(args.weights_dir)
    required_weights = [
        os.path.join(args.weights_dir, "profenet_v3_backbone.pt"),
        *[
            os.path.join(args.weights_dir, f"profenet_v3_table3_fold{fold}.pt")
            for fold in range(3)
        ],
    ]
    missing = [path for path in required_weights if not os.path.isfile(path)]
    if missing:
        raise SystemExit(
            "missing released ProFE-Net weight files:\n  " + "\n  ".join(missing)
        )

    pdbcode = os.path.splitext(os.path.basename(args.pdb))[0]
    workdir = args.cache_dir or tempfile.mkdtemp(prefix="profenet_predict_")
    csv_path, pdb_dir, complex_id = build_entry_csv(
        workdir, args.pdb, pdbcode, args.ligand_chains, args.receptor_chains, args.mutations
    )

    transform = get_transform(
        [
            {"type": "select_atom", "resolution": "backbone+CB"},
            {"type": "selected_region_fixed_size_patch", "select_attr": "mut_flag", "patch_size": 128},
        ]
    )
    dataset = SkempiDataset(
        csv_path=csv_path,
        pdb_dir=pdb_dir,
        cache_dir=os.path.join(workdir, "cache"),
        num_cvfolds=3,
        cvfold_index=2,
        split="val",
        fold_groups=[[], [], [complex_id]],
        fold_group_field="complex",
        transform=transform,
    )
    if len(dataset) != 1:
        raise RuntimeError(f"expected exactly one entry, got {len(dataset)} "
                           "(check chain letters and mutation format)")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate())
    batch = next(iter(loader))

    cfg = EasyDict(yaml.safe_load(open(args.config)))
    cfg.model.rde_checkpoint = required_weights[0]
    cfg.model.backbone_config = EasyDict(
        {
            "model": {
                "hidden_dim": 256,
                "pair_dim": 128,
                "n_heads": 16,
                "dropout": 0.1,
                "n_layers": 6,
                "ipa": {"depth": 2, "heads": 8, "dropout": 0.1},
            },
            # The released backbone contains the auxiliary structure head used
            # during pretraining; this flag reconstructs that exact module set.
            "training": {"aux_loss_weight": 1.0},
        }
    )
    device = torch.device(args.device)
    batch = recursive_to(batch, device)

    preds = []
    for fold in range(3):
        weights = os.path.join(args.weights_dir, f"profenet_v3_table3_fold{fold}.pt")
        model = model_factory(cfg).to(device)
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
        model.eval()
        with torch.no_grad():
            _, output = model(batch)
        preds.append(float(output["ddg"].detach().cpu().numpy()[0]))

    mean_pred = sum(preds) / len(preds)
    print(f"complex={complex_id} mutations={args.mutations}")
    for fold, value in enumerate(preds):
        print(f"  fold{fold} ddG_pred = {value:.4f} kcal/mol")
    print(f"  mean   ddG_pred = {mean_pred:.4f} kcal/mol")
    print("sign convention: ddG = dG_mutant - dG_wildtype; "
          "positive = weaker binding, negative = stronger binding")


if __name__ == "__main__":
    main()
