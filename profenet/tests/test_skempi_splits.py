import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rde.utils.skempi_splits import (
    audit_folds,
    balanced_pdb_folds,
    rde_complex_folds,
)


class SkempiSplitTest(unittest.TestCase):
    def setUp(self):
        self.entries = []
        sizes = [9, 7, 5, 3, 2, 2, 1]
        for pdb_index, size in enumerate(sizes):
            pdbcode = "P%d" % pdb_index
            for row in range(size):
                self.entries.append(
                    {
                        "pdbcode": pdbcode,
                        "complex": "%s_A_B" % pdbcode,
                        "id": "%s-%d" % (pdbcode, row),
                    }
                )

    def test_rde_protocol_is_deterministic(self):
        first = rde_complex_folds(self.entries, num_folds=3, seed=2022)
        second = rde_complex_folds(self.entries, num_folds=3, seed=2022)
        self.assertEqual(first, second)
        audit = audit_folds(self.entries, first, "complex")
        self.assertTrue(audit["all_entries_assigned_once"])
        self.assertFalse(audit["complex_overlap"])

    def test_balanced_protocol_is_pdb_disjoint(self):
        folds = balanced_pdb_folds(self.entries, num_folds=3, seed=2022)
        audit = audit_folds(self.entries, folds, "pdbcode")
        self.assertFalse(audit["pdb_overlap"])
        self.assertEqual(sum(audit["fold_rows"]), len(self.entries))
        self.assertLessEqual(
            max(audit["fold_rows"]) - min(audit["fold_rows"]),
            max([9, 7, 5, 3, 2, 2, 1]),
        )


if __name__ == "__main__":
    unittest.main()
