import os
import sys
import unittest

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rde.utils.transforms.corrupt_chi import CorruptChiAngle


class CorruptChiAngleTest(unittest.TestCase):
    def test_short_structure_matches_official_transform_contract(self):
        length = 9
        data = {
            "aa": torch.zeros(length, dtype=torch.long),
            "pos_atoms": torch.randn(length, 5, 3),
            "mask_atoms": torch.ones(length, 5, dtype=torch.bool),
            "chi": torch.zeros(length, 4),
            "chi_alt": torch.zeros(length, 4),
            "chi_mask": torch.ones(length, 4, dtype=torch.bool),
            "focus_flag": torch.ones(length, dtype=torch.bool),
        }
        transform = CorruptChiAngle(
            ratio_mask=0.1,
            add_noise=False,
            maskable_flag_attr="focus_flag",
        )

        transformed = transform(data)

        self.assertEqual(transformed["aa"].shape[0], length)
        self.assertEqual(int(transformed["chi_masked_flag"].sum()), 1)
        self.assertTrue(torch.isfinite(transformed["chi_native"]).all())
        self.assertTrue(torch.isfinite(transformed["chi_corrupt"]).all())


if __name__ == "__main__":
    unittest.main()
