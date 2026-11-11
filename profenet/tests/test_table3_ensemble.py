import os
import sys
import tempfile
import unittest

import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ensemble_table3_predictions import fixed_equal_ensemble


class Table3EnsembleTest(unittest.TestCase):
    def test_equal_ensemble_aligns_by_sample_id(self):
        first = pd.DataFrame(
            {
                "sample_id": [2, 1],
                "fold": [1, 0],
                "complex": ["B", "A"],
                "mutstr": ["WB2A", "WA1G"],
                "num_muts": [1, 1],
                "ddG": [2.0, 1.0],
                "ddG_pred": [3.0, 0.0],
            }
        )
        second = first.copy()
        second["ddG_pred"] = [1.0, 2.0]

        with tempfile.TemporaryDirectory() as directory:
            first_path = os.path.join(directory, "first.csv")
            second_path = os.path.join(directory, "second.csv")
            first.to_csv(first_path, index=False)
            second.iloc[::-1].to_csv(second_path, index=False)
            output = fixed_equal_ensemble(first_path, second_path, "fixed")

        self.assertEqual(output["sample_id"].tolist(), [1, 2])
        self.assertEqual(output["ddG_pred"].tolist(), [1.0, 2.0])

    def test_identity_mismatch_fails(self):
        row = {
            "sample_id": [1],
            "fold": [0],
            "complex": ["A"],
            "mutstr": ["WA1G"],
            "num_muts": [1],
            "ddG": [1.0],
            "ddG_pred": [0.0],
        }
        with tempfile.TemporaryDirectory() as directory:
            first_path = os.path.join(directory, "first.csv")
            second_path = os.path.join(directory, "second.csv")
            pd.DataFrame(row).to_csv(first_path, index=False)
            row["fold"] = [1]
            pd.DataFrame(row).to_csv(second_path, index=False)
            with self.assertRaisesRegex(ValueError, "fold"):
                fixed_equal_ensemble(first_path, second_path, "fixed")


if __name__ == "__main__":
    unittest.main()
