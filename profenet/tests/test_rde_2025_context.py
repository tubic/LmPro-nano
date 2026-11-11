import os
import sys
import unittest
from types import SimpleNamespace

import torch
import yaml
from easydict import EasyDict


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rde.models.rde_2025 import model_factory


def dict_to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: dict_to_namespace(item) for key, item in value.items()}
        )
    return value


class EnhancedRDEContextTest(unittest.TestCase):
    def test_chi_context_modes_are_finite_and_distinct(self):
        config_path = os.path.join(REPO_ROOT, "configs/train/rde_2025.yml")
        with open(config_path) as handle:
            raw = yaml.safe_load(handle)
        checkpoint = os.path.join(REPO_ROOT, raw["model"]["rde_checkpoint"])
        if not os.path.exists(checkpoint):
            self.skipTest("EnhancedRDE checkpoint is not available")

        torch.manual_seed(7)
        raw["model"]["head"]["rich"] = False
        raw["model"]["pool"]["mode"] = "global_max"
        model = model_factory(EasyDict(raw)).eval()
        self.assertEqual(model.pool_mode, "global_max")
        length = 8
        batch = {
            "aa": torch.randint(0, 20, (1, length)),
            "mask": torch.ones(1, length, dtype=torch.bool),
            "pos_heavyatom": torch.randn(1, length, 15, 3),
            "chi": torch.randn(1, length, 4),
            "mut_flag": torch.zeros(1, length, dtype=torch.bool),
        }
        batch["mut_flag"][:, 3] = True

        alanine_batch = {
            "aa": torch.zeros(1, 2, dtype=torch.long),
            "mask_heavyatom": torch.ones(1, 2, 15, dtype=torch.bool),
        }
        self.assertTrue(model._residue_mask(alanine_batch, alanine_batch["aa"]).all())

        padded_batch = {key: value.clone() for key, value in batch.items()}
        padded_batch["aa"][:, -1] = 21
        padded_batch["mask"][:, -1] = False
        padded_batch["pos_heavyatom"][:, -1] = 0
        safe_padded_batch = {key: value.clone() for key, value in padded_batch.items()}
        safe_padded_batch["aa"][:, -1] = 0
        with torch.no_grad():
            padded_context = model._get_rde_context(padded_batch)
            safe_padded_context = model._get_rde_context(safe_padded_batch)
        self.assertTrue(torch.isfinite(padded_context).all())
        self.assertTrue(torch.allclose(padded_context, safe_padded_context, atol=1e-6))

        # Native SkempiDataset exposes WT residues as `aa`, without the
        # adapter-only `aa_wt` alias used by older training scripts.
        full_batch = {
            **batch,
            "aa_mut": batch["aa"].clone(),
            "mask_heavyatom": torch.ones(
                1, length, 15, dtype=torch.bool
            ),
            "chain_nb": torch.zeros(1, length, dtype=torch.long),
            "res_nb": torch.arange(length).unsqueeze(0),
            "phi": torch.randn(1, length),
            "psi": torch.randn(1, length),
            "phi_mask": torch.ones(1, length, dtype=torch.bool),
            "psi_mask": torch.ones(1, length, dtype=torch.bool),
            "chi_mask": torch.ones(1, length, 4, dtype=torch.bool),
            "ddG": torch.tensor([0.5]),
        }
        full_batch["aa_mut"][:, 3] = (full_batch["aa"][:, 3] + 1) % 20
        losses, output = model(full_batch)
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        self.assertTrue(torch.isfinite(output["ddg"]).all())

        contexts = {}
        with torch.no_grad():
            for mode in ("none", "native", "mask_mut"):
                model.backbone_chi_mode = mode
                contexts[mode] = model._get_rde_context(batch)

        for context in contexts.values():
            self.assertEqual(tuple(context.shape), (1, length, 256))
            self.assertTrue(torch.isfinite(context).all())
        self.assertFalse(torch.allclose(contexts["none"], contexts["native"]))
        self.assertFalse(torch.allclose(contexts["native"], contexts["mask_mut"]))

        masked_hint = model.rde.prepare_chi_hint(
            batch["chi"],
            masked_flag=batch["mut_flag"],
            apply_dropout=False,
        )
        self.assertTrue(torch.equal(masked_hint[:, 3], torch.zeros_like(masked_hint[:, 3])))


if __name__ == "__main__":
    unittest.main()
