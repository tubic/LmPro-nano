#!/usr/bin/env python3
"""Export release weights from the locked training checkpoints.

Strips optimizer/scheduler/RNG state and writes plain model state_dicts plus a
provenance manifest. Run from the repository root:

    python scripts/export_release_weights.py release/profenet_20260810/weights
"""

import hashlib
import json
import os
import sys

import torch

PROFE_RUN = "outputs/table3_runs/table3-profe_enhanced-rde3_2026_08_07__19_32_40_profenet_v3_96k_full"
RDE_RUN = "outputs/table3_runs/table3-rde_profe-rde3_2026_08_02__21_22_39_best191k_full_bs16_acc2"
PROFE_BACKBONE = (
    "outputs/profenet_pretrain_v3/"
    "profenet_2026_08_05__14_47_02_skempi50_disjoint_maskfocus_v3_from_scratch/"
    "best_backbone.pt"
)
RDE_BACKBONE = "outputs/rde_pretrain/rde_2026_07_28__01_30_12_full_rde_table3_local_cache/checkpoints/best.pt"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(state_dict, dest):
    torch.save(state_dict, dest)
    return sha256_file(dest)


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    provenance = {"schema_version": 1, "artifacts": []}

    def record(name, source, source_sha256, note):
        provenance["artifacts"].append(
            {
                "file": name,
                "sha256": sha256_file(os.path.join(outdir, name)),
                "source_checkpoint": source,
                "source_checkpoint_sha256": source_sha256,
                "note": note,
            }
        )
        print(f"wrote {name}")

    # ProFE-Net v3 pretrained backbone (iteration 96,000, pretraining-validation
    # selection only; source already stores a bare backbone state dict).
    src = PROFE_BACKBONE
    blob = torch.load(src, map_location="cpu", weights_only=False)
    save(blob["model_state_dict"], os.path.join(outdir, "profenet_v3_backbone.pt"))
    record(
        "profenet_v3_backbone.pt",
        src,
        sha256_file(src),
        "ProFE-Net v3 pretrained EnhancedRDE backbone, iteration "
        f"{blob.get('iteration')}, best_val_loss={blob.get('best_val_loss')}",
    )

    # ProFE-Net v3 downstream fold models (fixed 10,000 updates per fold).
    src = os.path.join(PROFE_RUN, "checkpoints", "last.pt")
    src_sha = sha256_file(src)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    assert len(ckpt["models"]) == 3
    for fold, state in enumerate(ckpt["models"]):
        name = f"profenet_v3_table3_fold{fold}.pt"
        save(state, os.path.join(outdir, name))
        record(name, src, src_sha, f"ProFE-Net v3 Table 3 fold {fold} head+backbone after 10,000 locked updates")

    # Local RDE pretrained backbone (best.pt, 'best191k' run).
    src = RDE_BACKBONE
    blob = torch.load(src, map_location="cpu", weights_only=False)
    save(blob["model"], os.path.join(outdir, "rde_backbone.pt"))
    record(
        "rde_backbone.pt",
        src,
        sha256_file(src),
        f"Locally pretrained RDE backbone, iteration {blob.get('iteration')}",
    )

    # Local RDE downstream fold models.
    src = os.path.join(RDE_RUN, "checkpoints", "last.pt")
    src_sha = sha256_file(src)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    assert len(ckpt["models"]) == 3
    for fold, state in enumerate(ckpt["models"]):
        name = f"rde_table3_fold{fold}.pt"
        save(state, os.path.join(outdir, name))
        record(name, src, src_sha, f"Local RDE Table 3 fold {fold} model after 10,000 locked updates")

    with open(os.path.join(outdir, "weights_provenance.json"), "w") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)

    with open(os.path.join(outdir, "SHA256SUMS"), "w") as handle:
        for artifact in sorted(provenance["artifacts"], key=lambda a: a["file"]):
            handle.write(f"{artifact['sha256']}  {artifact['file']}\n")
    print("weights export completed")


if __name__ == "__main__":
    main()
