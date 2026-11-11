import os
import sys
import unittest
from types import SimpleNamespace

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rde.models.rde_profe import model_factory


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
    return value


class ProFERDENetworkTest(unittest.TestCase):
    def test_smoke_checkpoint_forward_backward_and_antisymmetry(self):
        checkpoint = os.environ.get(
            "RDE_TEST_CHECKPOINT",
            "outputs/rde_pretrain/rde-resume_2026_07_27__11_18_48_gpu_smoke_resume/checkpoints/best.pt",
        )
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(REPO_ROOT, checkpoint)
        if not os.path.exists(checkpoint):
            self.skipTest("RDE smoke checkpoint is not available")
        cfg = namespace(
            {
                "model": {
                    "rde_checkpoint": checkpoint,
                    "backbone_dim": 128,
                    "pooling": "max",
                    "antisymmetric_output": True,
                    "encoder": {
                        "node_feat_dim": 128,
                        "pair_feat_dim": 64,
                        "num_layers": 3,
                    },
                }
            }
        )
        torch.manual_seed(11)
        model = model_factory(cfg).eval()
        batch_size, length = 2, 8
        aa_wt = torch.randint(0, 20, (batch_size, length))
        aa_mut = aa_wt.clone()
        aa_mut[:, 3] = (aa_mut[:, 3] + 1) % 20
        batch = {
            "aa_wt": aa_wt,
            "aa_mut": aa_mut,
            "pos_heavyatom": torch.randn(batch_size, length, 15, 3),
            "mask_heavyatom": torch.ones(batch_size, length, 15, dtype=torch.bool),
            "mask": torch.ones(batch_size, length, dtype=torch.bool),
            "chain_nb": torch.zeros(batch_size, length, dtype=torch.long),
            "res_nb": torch.arange(length).repeat(batch_size, 1),
            "mut_flag": torch.zeros(batch_size, length, dtype=torch.bool),
            "phi": torch.randn(batch_size, length),
            "psi": torch.randn(batch_size, length),
            "chi": torch.randn(batch_size, length, 4),
            "phi_mask": torch.ones(batch_size, length, dtype=torch.bool),
            "psi_mask": torch.ones(batch_size, length, dtype=torch.bool),
            "chi_mask": torch.ones(batch_size, length, 4, dtype=torch.bool),
            "ddG": torch.tensor([0.7, -1.1]),
        }
        batch["mut_flag"][:, 3] = True

        losses, output = model(batch)
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        self.assertTrue(torch.isfinite(output["ddg"]).all())
        losses["total_loss"].backward()
        self.assertTrue(any(p.grad is not None for p in model.ddg_readout.parameters()))

        reverse = dict(batch)
        reverse["aa_wt"], reverse["aa_mut"] = batch["aa_mut"], batch["aa_wt"]
        reverse["ddG"] = -batch["ddG"]
        with torch.no_grad():
            reverse_prediction = model(reverse)[1]["ddg"]
        self.assertTrue(torch.allclose(output["ddg"], -reverse_prediction, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
