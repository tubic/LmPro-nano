import json
import os
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from nested_cv_select import evaluate_locked_selection, lock_inner_dev_selection


class NestedCVSelectionTest(unittest.TestCase):
    def test_selection_is_inner_dev_only_and_folds_are_disjoint(self):
        with tempfile.TemporaryDirectory() as root:
            scores = {"A": [0.8, 0.2], "B": [0.3, 0.9]}
            targets = [
                np.array([-2.0, -0.5, 0.5, 1.5]),
                np.array([-1.5, -0.2, 0.7, 2.0]),
            ]
            groups = [["P0"] * 4, ["P1"] * 4]
            for tag in scores:
                for fold in (0, 1):
                    directory = os.path.join(root, "ddg_%s_fold%d" % (tag, fold))
                    os.makedirs(directory)
                    with open(os.path.join(directory, "selection.json"), "w") as handle:
                        json.dump(
                            {
                                "selected_epoch": 3,
                                "inner_dev_spearman": scores[tag][fold],
                                "selection_scope": "inner-dev",
                                "outer_predictions_opened": False,
                            },
                            handle,
                        )
                    prediction = targets[fold] + (0.1 if tag == "A" else -0.1)
                    np.save(
                        os.path.join(directory, "best_preds.npy"),
                        np.stack([prediction, targets[fold]]),
                    )
                    with open(os.path.join(directory, "best_groups.json"), "w") as handle:
                        json.dump(groups[fold], handle)

            selection = lock_inner_dev_selection(root, ["A", "B"], [0, 1])
            self.assertEqual(
                [record["selected_tag"] for record in selection["folds"]],
                ["A", "B"],
            )
            self.assertTrue(selection["reviewer_grade"])
            result = evaluate_locked_selection(root, selection)
            self.assertTrue(result["outer_fold_structure_disjoint"])
            self.assertEqual(result["pooled"]["n"], 8)


if __name__ == "__main__":
    unittest.main()
