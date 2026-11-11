import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rde.datasets.skempi import SkempiDataset
from rde.utils.skempi_splits import balanced_pdb_folds, rde_complex_folds


class SkempiDatasetProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = os.path.join(REPO_ROOT, "data/SKEMPI_v2/skempi_v2.csv")
        cls.pdb_dir = os.path.join(REPO_ROOT, "data/SKEMPI_v2/PDBs")
        cls.cache_dir = os.path.join(
            REPO_ROOT, "data/SKEMPI_v2_cache_rde_protocol"
        )
        if not os.path.exists(cls.csv_path):
            raise unittest.SkipTest("SKEMPI data is unavailable")
        cls.reference = SkempiDataset(
            csv_path=cls.csv_path,
            pdb_dir=cls.pdb_dir,
            cache_dir=cls.cache_dir,
            num_cvfolds=3,
            cvfold_index=0,
            split="train",
        )
        cls.entries = cls.reference.entries_full

    def assert_protocol(self, folds, group_field):
        all_ids = {int(entry["id"]) for entry in self.entries}
        validation_ids = []
        validation_groups = []
        for fold in range(len(folds)):
            common = {
                "csv_path": self.csv_path,
                "pdb_dir": self.pdb_dir,
                "cache_dir": self.cache_dir,
                "num_cvfolds": len(folds),
                "cvfold_index": fold,
                "fold_groups": folds,
                "fold_group_field": group_field,
            }
            train = SkempiDataset(split="train", **common)
            validation = SkempiDataset(split="val", **common)
            train_ids = {int(entry["id"]) for entry in train.entries}
            fold_ids = {int(entry["id"]) for entry in validation.entries}
            self.assertFalse(train_ids & fold_ids)
            self.assertEqual(train_ids, all_ids - fold_ids)
            validation_ids.append(fold_ids)
            validation_groups.append(
                {str(entry[group_field]) for entry in validation.entries}
            )

        self.assertEqual(set().union(*validation_ids), all_ids)
        self.assertEqual(sum(map(len, validation_ids)), len(all_ids))
        for left in range(len(folds)):
            for right in range(left + 1, len(folds)):
                self.assertFalse(validation_groups[left] & validation_groups[right])

    def test_explicit_rde3_matches_one_time_complex_assignment(self):
        folds = rde_complex_folds(self.entries, num_folds=3, seed=2022)
        self.assert_protocol(folds, "complex")

    def test_balanced_pdb5_is_one_time_pdb_assignment(self):
        folds = balanced_pdb_folds(self.entries, num_folds=5, seed=2022)
        self.assert_protocol(folds, "pdbcode")


if __name__ == "__main__":
    unittest.main()
