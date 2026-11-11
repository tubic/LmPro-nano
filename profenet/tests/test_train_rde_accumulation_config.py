import os
import sys
import unittest
from types import SimpleNamespace

from easydict import EasyDict


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from train_rde import apply_overrides


def arguments(accumulation_steps):
    return SimpleNamespace(
        max_iters=None,
        batch_size=32,
        accumulation_steps=accumulation_steps,
        processed_dir=None,
        val_freq=None,
        num_workers=4,
        max_val_batches=None,
        save_freq=1000,
    )


class RDEAccumulationConfigTest(unittest.TestCase):
    def test_override_preserves_effective_batch(self):
        config = EasyDict({"train": {"batch_size": 64}})
        apply_overrides(config, arguments(accumulation_steps=2))
        self.assertEqual(config.train.batch_size, 32)
        self.assertEqual(config.train.accumulation_steps, 2)
        self.assertEqual(
            config.train.batch_size * config.train.accumulation_steps, 64
        )

    def test_missing_value_defaults_to_one(self):
        config = EasyDict({"train": {"batch_size": 64}})
        args = arguments(accumulation_steps=None)
        args.batch_size = None
        apply_overrides(config, args)
        self.assertEqual(config.train.accumulation_steps, 1)

    def test_nonpositive_accumulation_is_rejected(self):
        config = EasyDict({"train": {"batch_size": 64}})
        with self.assertRaisesRegex(ValueError, "accumulation-steps"):
            apply_overrides(config, arguments(accumulation_steps=0))

    def test_processed_dir_overrides_both_splits(self):
        config = EasyDict(
            {
                "train": {"batch_size": 64},
                "data": {
                    "train": {"processed_dir": "old-train"},
                    "val": {"processed_dir": "old-val"},
                },
            }
        )
        args = arguments(accumulation_steps=1)
        args.processed_dir = "/tmp/rde-cache"

        apply_overrides(config, args)

        self.assertEqual(config.data.train.processed_dir, "/tmp/rde-cache")
        self.assertEqual(config.data.val.processed_dir, "/tmp/rde-cache")


if __name__ == "__main__":
    unittest.main()
