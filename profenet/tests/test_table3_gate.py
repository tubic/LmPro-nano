import os
import sys
import tempfile
import unittest

import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from check_table3_metrics import check_metrics, load_thresholds


class Table3GateTest(unittest.TestCase):
    def test_all_thresholds_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "metrics.csv")
            row = {
                "mode": "single",
                "overall_pearson": 0.64,
                "overall_spearman": 0.54,
                "rmse": 1.35,
                "mae": 0.95,
                "auroc": 0.76,
            }
            pd.DataFrame([row]).to_csv(path, index=False)
            self.assertTrue(check_metrics(path)["passed"])

            row["auroc"] = 0.74
            pd.DataFrame([row]).to_csv(path, index=False)
            result = check_metrics(path)
            self.assertFalse(result["passed"])
            self.assertFalse(result["checks"]["auroc"]["passed"])

    def test_versioned_superiority_thresholds(self):
        threshold_path = os.path.join(
            REPO_ROOT,
            "configs",
            "train",
            "profenet_rde_superiority_gate.json",
        )
        thresholds = load_thresholds(threshold_path)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "metrics.csv")
            manuscript_only = {
                "mode": "single",
                "overall_pearson": 0.6365,
                "overall_spearman": 0.5344,
                "rmse": 1.3652,
                "mae": 0.9617,
                "auroc": 0.7523,
            }
            pd.DataFrame([manuscript_only]).to_csv(path, index=False)
            self.assertFalse(check_metrics(path, thresholds=thresholds)["passed"])

            beats_both = {
                "mode": "single",
                "overall_pearson": 0.6464,
                "overall_spearman": 0.5344,
                "rmse": 1.3270,
                "mae": 0.9391,
                "auroc": 0.7523,
            }
            pd.DataFrame([beats_both]).to_csv(path, index=False)
            self.assertTrue(check_metrics(path, thresholds=thresholds)["passed"])


if __name__ == "__main__":
    unittest.main()
