import os
import sys
import unittest

import numpy as np
import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rde.utils.skempi import overall_rmse_mae


class SkempiMetricsTest(unittest.TestCase):
    def test_published_calibration_and_raw_errors_are_both_reported(self):
        true = np.array([-2.0, -1.0, 1.0, 2.0])
        prediction = true * 0.5 + 3.0
        metrics = overall_rmse_mae(
            pd.DataFrame({"ddG": true, "ddG_pred": prediction})
        )

        self.assertAlmostEqual(metrics["rmse"], 0.0, places=6)
        self.assertAlmostEqual(metrics["mae"], 0.0, places=6)
        self.assertGreater(metrics["rmse_raw"], 2.0)
        self.assertGreater(metrics["mae_raw"], 2.0)
        self.assertAlmostEqual(metrics["calibration_slope"], 2.0, places=6)
        self.assertAlmostEqual(metrics["calibration_intercept"], -6.0, places=6)


if __name__ == "__main__":
    unittest.main()
