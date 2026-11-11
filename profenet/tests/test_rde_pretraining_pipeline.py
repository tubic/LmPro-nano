import math
import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from rde.datasets.pdbredo_chain import get_pdbredo_chain_dataset
from rde.models.rde import CircularSplineRotamerDensityEstimator
from rde.utils.data import PaddingCollate
from rde.utils.misc import load_config, seed_all


class RDEPretrainingPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = load_config("configs/train/rde.yml")
        processed_dir = os.environ.get("RDE_PROCESSED_DIR")
        if processed_dir:
            cls.config.data.train.processed_dir = processed_dir
            cls.config.data.val.processed_dir = processed_dir
        seed_all(cls.config.train.seed)
        cls.train = get_pdbredo_chain_dataset(cls.config.data.train)
        cls.val = get_pdbredo_chain_dataset(cls.config.data.val)

    def test_cluster_splits_are_nonempty_and_disjoint(self):
        self.assertGreater(len(self.train), 1_000)
        self.assertGreater(len(self.val), 10)
        self.assertTrue(self.train.sample_by_cluster)
        self.assertTrue(self.val.sample_by_cluster)
        self.assertFalse(
            set(self.train._clusters_of_split) & set(self.val._clusters_of_split)
        )

    def test_sample_matches_official_rde_contract(self):
        sample = self.train[0]
        required = {
            "aa",
            "pos_atoms",
            "mask_atoms",
            "chi_native",
            "chi_corrupt",
            "chi_masked_flag",
            "chi_corrupt_flag",
            "focus_flag",
        }
        self.assertTrue(required.issubset(sample))
        self.assertLessEqual(sample["aa"].shape[0], 128)
        self.assertEqual(sample["pos_atoms"].shape[1], 5)
        self.assertEqual(sample["focus_flag"].shape[0], sample["aa"].shape[0])

        # Both splits share one on-disk LMDB cache.  This catches accidental
        # duplicate environment opens in the single-process smoke-test mode.
        val_sample = self.val[0]
        self.assertTrue(required.issubset(val_sample))

    def test_model_forward_and_backward_are_finite(self):
        batch = PaddingCollate()([self.train[0]])
        model = CircularSplineRotamerDensityEstimator(self.config.model)
        losses = model(batch)
        total = sum(
            float(self.config.train.loss_weights[name]) * value
            for name, value in losses.items()
        )
        self.assertTrue(math.isfinite(float(total.detach())))
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_worker_processes_reopen_shared_lmdb(self):
        collate = PaddingCollate()
        train_loader = DataLoader(
            self.train, batch_size=1, num_workers=2, collate_fn=collate
        )
        val_loader = DataLoader(
            self.val, batch_size=1, num_workers=2, collate_fn=collate
        )
        self.assertEqual(next(iter(train_loader))["size"], 1)
        self.assertEqual(next(iter(val_loader))["size"], 1)


if __name__ == "__main__":
    unittest.main()
