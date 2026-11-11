import os
import sys
import tempfile
import unittest

import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from summarize_table3_uncertainty import align_reference, bootstrap, metric_values


class Table3UncertaintyTest(unittest.TestCase):
    def make_frame(self, offset=0.0):
        rows = []
        sample_id = 0
        for fold in range(2):
            for complex_index in range(3):
                for value in (-2.0, -0.5, 0.5, 2.0):
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "fold": fold,
                            "complex": f"f{fold}-c{complex_index}",
                            "mutstr": f"A{sample_id}V",
                            "num_muts": 1,
                            "ddG": value,
                            "method": "candidate",
                            "ddG_pred": value + offset,
                        }
                    )
                    sample_id += 1
        return pd.DataFrame(rows)

    def test_cluster_bootstrap_is_deterministic_and_paired(self):
        candidate = self.make_frame(offset=0.0)
        reference = self.make_frame(offset=0.4)
        reference["method"] = "reference"
        aligned = align_reference(candidate, reference)

        first = bootstrap(aligned, 20, seed=7, confidence=0.95, paired=True)
        second = bootstrap(aligned, 20, seed=7, confidence=0.95, paired=True)
        self.assertEqual(first, second)
        self.assertAlmostEqual(metric_values(candidate)["overall_pearson"], 1.0)
        self.assertIsNotNone(first[2]["mae_raw"]["ci_low"])

    def test_alignment_rejects_changed_identity(self):
        candidate = self.make_frame()
        reference = self.make_frame()
        reference.loc[0, "complex"] = "wrong"
        with self.assertRaisesRegex(ValueError, "complex"):
            align_reference(candidate, reference)


if __name__ == "__main__":
    unittest.main()
