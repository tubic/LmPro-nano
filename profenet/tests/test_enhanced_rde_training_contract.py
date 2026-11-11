import math
import io
import os
import random
import sys
import unittest

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rde.models.enhanced_rde import EnhancedRDE, OptimizedAxialAttention
from rde.datasets.pdbredo_chain import PDBRedoChainAdapter
from rde.utils.chi_context import build_chi_target_mask, encode_chi_hint
from rde.utils.chi_symmetry import CHI_SYMMETRY_PERIODS
from rde.utils.protein.constants import AA, BBHeavyAtom
from evaluate_enhanced_rde import set_evaluation_seed
from train_enhanced_rde import (
    calculate_loss,
    custom_collate,
    make_backbone_artifact,
    validate_batch_contract,
    validate_config_contract,
)


def tiny_config():
    return {
        "model": {
            "hidden_dim": 32,
            "pair_dim": 16,
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
            "num_recycle": 0,
            "aux_layer_idx": 0,
            "use_chi_hint": True,
            "chi_hint_dropout_prob": 0.0,
            "gnn": {
                "layers": 1,
                "v_dim": 4,
                "use_radius": False,
                "radius": 12.0,
                "k_neighbors": 4,
            },
            "ipa": {"depth": 1, "heads": 4, "dropout": 0.0},
        },
        "training": {
            "aux_loss_weight": 0.5,
            "loss_weights": {
                "chi_prediction_weights": [3.0, 2.0, 0.5, 0.5],
                "symmetry_aware": True,
                "chi_target": "all",
                "masked_chi_aux_weight": 0.0,
                "denoising": 1.0,
            },
        },
    }


def protein_sample(aa):
    aa = torch.as_tensor(aa, dtype=torch.long)
    length = int(aa.numel())
    positions = torch.zeros(length, 15, 3)
    atom_mask = torch.zeros(length, 15, dtype=torch.bool)
    for residue in range(length):
        origin = torch.tensor([3.8 * residue, 0.2 * (residue % 2), 0.0])
        positions[residue, BBHeavyAtom.N] = origin + torch.tensor([-1.2, 0.2, 0.0])
        positions[residue, BBHeavyAtom.CA] = origin
        positions[residue, BBHeavyAtom.C] = origin + torch.tensor([1.3, 0.5, 0.1])
        positions[residue, BBHeavyAtom.O] = origin + torch.tensor([1.8, 1.4, 0.1])
        positions[residue, BBHeavyAtom.CB] = origin + torch.tensor([-0.2, -1.1, 1.0])
        atom_mask[residue, :5] = True
    chi_mask = torch.zeros(length, 4, dtype=torch.bool)
    chi_mask[aa != int(AA.ALA), 0] = True
    return {
        "id": "sample",
        "aa": aa,
        "pos_heavyatom": positions.clone(),
        "pos_atoms": positions.clone(),
        "pos_atoms_native": positions.clone(),
        "mask_heavyatom": atom_mask.clone(),
        "mask_atoms": atom_mask,
        "chi_native": torch.zeros(length, 4),
        "chi_corrupt": torch.zeros(length, 4),
        "chi_mask": chi_mask,
        "chi_corrupt_flag": torch.ones(length, dtype=torch.bool),
        "chi_masked_flag": torch.zeros(length, dtype=torch.bool),
    }


class EnhancedRDETrainingContractTest(unittest.TestCase):
    def test_evaluation_seed_replays_all_rng_streams(self):
        set_evaluation_seed(1234)
        first = (random.random(), np.random.rand(), torch.rand(3))
        set_evaluation_seed(1234)
        second = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_adapter_can_skip_unused_precomputed_features(self):
        sample = protein_sample([AA.ALA, AA.CYS])

        class BaseDataset:
            def __getitem__(self, _index):
                return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in sample.items()}

        adapter = object.__new__(PDBRedoChainAdapter)
        adapter.base_dataset = BaseDataset()
        adapter.processed_dir = "/missing/features/are/not/read"
        adapter.transform = None
        adapter.load_precomputed_features = False
        adapter.valid_indices = [0]
        loaded = adapter[0]
        self.assertNotIn("geometry_features", loaded)
        self.assertNotIn("dihedral_features", loaded)
        self.assertNotIn("local_env_features", loaded)
        self.assertIn("pos_atoms", loaded)

    def test_config_rejects_enabled_denoising_without_noise_transform(self):
        config = tiny_config()
        config["data"] = {
            "transform": [{"type": "corrupt_chi_angle"}],
            "validation": {"transform": [{"type": "corrupt_chi_angle"}]},
        }
        with self.assertRaisesRegex(ValueError, "add_atom_noise"):
            validate_config_contract(config)

    def test_config_rejects_invalid_chi_target_and_hint_dropout(self):
        transforms = [
            {"type": "add_atom_noise"},
            {"type": "corrupt_chi_angle"},
        ]
        config = tiny_config()
        config["data"] = {
            "transform": transforms,
            "validation": {"transform": transforms},
        }
        config["training"]["loss_weights"]["chi_target"] = "unknown"
        with self.assertRaisesRegex(ValueError, "chi_target"):
            validate_config_contract(config)

        config["training"]["loss_weights"]["chi_target"] = "corrupt"
        config["model"]["chi_hint_dropout_prob"] = 1.0
        with self.assertRaisesRegex(ValueError, "chi_hint_dropout_prob"):
            validate_config_contract(config)

    def test_collate_uses_lengths_and_preserves_alanine(self):
        first = protein_sample([AA.ALA, AA.CYS, AA.ALA])
        second = protein_sample([AA.ALA])
        first["id"] = "mixed"
        second["id"] = "all_ala"
        batch = custom_collate([first, second])

        self.assertEqual(batch["mask"].tolist(), [[True, True, True], [True, False, False]])
        self.assertEqual(int(batch["aa"][0, 0]), int(AA.ALA))
        self.assertEqual(int(batch["aa"][1, 0]), int(AA.ALA))
        self.assertEqual(int(batch["aa"][1, 1]), int(AA.UNK))
        validate_batch_contract(batch)

    def test_legacy_zero_value_mask_rejects_all_alanine_chain(self):
        batch = custom_collate([protein_sample([AA.ALA] * 4)])
        batch["mask"] = batch["aa"] != 0
        with self.assertRaisesRegex(ValueError, "empty residue masks"):
            validate_batch_contract(batch)

    def test_symmetry_periods_match_project_vocabulary(self):
        self.assertEqual(
            CHI_SYMMETRY_PERIODS,
            {
                int(AA.ASP): {2: math.pi},
                int(AA.GLU): {3: math.pi},
                int(AA.PHE): {2: math.pi},
                int(AA.TYR): {2: math.pi},
            },
        )
        self.assertNotIn(int(AA.UNK), CHI_SYMMETRY_PERIODS)

    def test_chi_target_mask_selects_only_corrupted_or_masked_entries(self):
        sample = protein_sample([AA.CYS, AA.ASP, AA.GLU])
        sample["chi_corrupt_flag"] = torch.tensor([True, False, True])
        sample["chi_masked_flag"] = torch.tensor([False, False, True])
        batch = custom_collate([sample])

        corrupt = build_chi_target_mask(batch, "corrupt")
        masked = build_chi_target_mask(batch, "masked")
        self.assertTrue(corrupt[0, 0, 0])
        self.assertFalse(corrupt[0, 1].any())
        self.assertTrue(masked[0, 2, 0])
        self.assertFalse(masked[0, :2].any())

    def test_masked_chi_hint_is_not_confused_with_zero_radians(self):
        angles = torch.zeros(1, 2, 4)
        chi_mask = torch.ones_like(angles, dtype=torch.bool)
        encoded = encode_chi_hint(
            angles,
            chi_mask=chi_mask,
            masked_flag=torch.tensor([[False, True]]),
        )
        self.assertEqual(encoded[0, 0, :2].tolist(), [0.0, 1.0])
        self.assertTrue(torch.equal(encoded[0, 1], torch.zeros(8)))

    def test_corrupt_target_loss_ignores_unflagged_chi(self):
        sample = protein_sample([AA.CYS, AA.CYS])
        sample["chi_corrupt_flag"] = torch.tensor([True, False])
        batch = custom_collate([sample])
        predictions = torch.zeros(1, 2, 4, 2)
        predictions[..., 1] = 1.0
        coordinates = torch.zeros(1, 2, 15, 3)
        weights = dict(tiny_config()["training"]["loss_weights"])
        weights["chi_target"] = "corrupt"

        reference = calculate_loss(
            (predictions, coordinates), batch, torch.device("cpu"), weights
        )[1]
        predictions[:, 1, 0] = torch.tensor([0.0, -1.0])
        changed_unflagged = calculate_loss(
            (predictions, coordinates), batch, torch.device("cpu"), weights
        )[1]
        self.assertTrue(torch.equal(reference, changed_unflagged))

        predictions[:, 0, 0] = torch.tensor([0.0, -1.0])
        changed_flagged = calculate_loss(
            (predictions, coordinates), batch, torch.device("cpu"), weights
        )[1]
        self.assertGreater(float(changed_flagged), float(reference))

    def test_masked_losses_do_not_read_nonfinite_padding(self):
        batch = custom_collate([protein_sample([AA.CYS]), protein_sample([AA.CYS, AA.CYS])])
        chi = torch.zeros(2, 2, 4, 2)
        chi[..., 1] = 1.0
        coord = torch.zeros(2, 2, 15, 3)
        chi[0, 1] = float("nan")
        coord[0, 1] = float("nan")
        loss, chi_loss, denoise_loss = calculate_loss(
            (chi, coord),
            batch,
            torch.device("cpu"),
            tiny_config()["training"]["loss_weights"],
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(chi_loss))
        self.assertTrue(torch.isfinite(denoise_loss))

    def test_all_false_axial_attention_is_finite_and_zero(self):
        module = OptimizedAxialAttention(16, 32, heads=4, dim_head=8)
        output = module(torch.randn(3, 5, 16), torch.zeros(3, 5, dtype=torch.bool))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_all_alanine_forward_backward_is_finite(self):
        torch.manual_seed(7)
        config = tiny_config()
        model = EnhancedRDE(config).train()
        batch = custom_collate([protein_sample([AA.ALA] * 5)])
        validate_batch_contract(batch)
        final_predictions, auxiliary_predictions = model(batch)
        for predictions in (final_predictions, auxiliary_predictions):
            self.assertTrue(all(torch.isfinite(value).all() for value in predictions))
        final_loss = calculate_loss(
            final_predictions,
            batch,
            torch.device("cpu"),
            config["training"]["loss_weights"],
        )[0]
        auxiliary_loss = calculate_loss(
            auxiliary_predictions,
            batch,
            torch.device("cpu"),
            config["training"]["loss_weights"],
        )[0]
        loss = final_loss + 0.5 * auxiliary_loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_valid_outputs_are_padding_invariant(self):
        torch.manual_seed(11)
        model = EnhancedRDE(tiny_config()).eval()
        sample = protein_sample([AA.ALA, AA.CYS, AA.ASP, AA.GLU])
        short = custom_collate([sample])
        longer = protein_sample([AA.ALA, AA.CYS, AA.ASP, AA.GLU, AA.PHE, AA.GLY])
        padded = custom_collate([longer])
        for key, value in short.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                padded[key][:, :4] = value
        padded["aa"][:, 4:] = int(AA.UNK)
        padded["mask"][:, 4:] = False
        for key in ("mask_atoms", "mask_heavyatom", "chi_mask"):
            padded[key][:, 4:] = False
        for key in (
            "pos_heavyatom",
            "pos_atoms",
            "pos_atoms_native",
            "chi_native",
            "chi_corrupt",
        ):
            padded[key][:, 4:] = 0

        with torch.no_grad():
            short_predictions = model(short)[0]
            padded_predictions = model(padded)[0]
        for short_value, padded_value in zip(short_predictions, padded_predictions):
            self.assertTrue(
                torch.allclose(short_value[:, :4], padded_value[:, :4], atol=1e-5, rtol=1e-5)
            )

    def test_geometry_encoder_reads_corrupted_coordinates(self):
        torch.manual_seed(13)
        model = EnhancedRDE(tiny_config()).eval()
        batch = custom_collate([protein_sample([AA.CYS, AA.ASP, AA.GLU])])
        batch["pos_atoms"][:, 1, BBHeavyAtom.CA, 1] += 2.0
        captured = {}

        def capture_input(_module, args):
            captured["positions"] = args[1].detach().clone()

        handle = model.geometry_encoder.register_forward_pre_hook(capture_input)
        with torch.no_grad():
            model(batch)
        handle.remove()
        expected = batch["pos_atoms"][:, :, BBHeavyAtom.CA].reshape(-1, 3)
        self.assertTrue(torch.equal(captured["positions"], expected))

    def test_backbone_artifact_is_weights_only_loadable(self):
        model = EnhancedRDE(tiny_config())
        payload = make_backbone_artifact(
            3,
            model,
            1.25,
            (20.0, 70.0, 80.0),
            tiny_config(),
        )
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=True)
        self.assertEqual(loaded["iteration"], 3)
        self.assertEqual(loaded["model_state_dict"].keys(), model.state_dict().keys())


if __name__ == "__main__":
    unittest.main()
