import argparse
import unittest

import torch
from fairseq.data import Dictionary

from inference import compute_structure_metrics
from tools.backfill_lmpro_source_lines import backfill_rows
from tools.export_proteinmpnn_benchmark import linearize_cdr_coordinates
from tools.merge_proteinmpnn_metrics import merge_payloads, merge_per_record_rows
from tools.paired_aar_bootstrap import bootstrap_comparison
from tools.summarize_lmpro_structure import summarize_rows

from fairseq_models.modules.loss_normalization import (
    combine_generation_loss,
    normalize_refinement_losses,
    token_weighted_structure_mean,
)
from fairseq_models.modules.ab_multihead_attention import MultiheadAttention
from fairseq_models.modules.protein_features import ProteinFeatures
from fairseq_models.modules.utils import compute_rmsd, full_square_dist, kabsch
from fairseq_models.criterions.abbert_masked_lm import AntibodyMaskedLmLoss
from fairseq_models.tasks.abbert_mask_tokens_dataset import AntibodyMaskTokensDataset
from fairseq_models.models.ab_roberta_base import AntibodyRobertaModel


class TrainingSemanticsTest(unittest.TestCase):
    def setUp(self):
        self.seq_dict = Dictionary()
        for token in ["A", "C", "D", "E", "F", "<mask>"]:
            self.seq_dict.add_symbol(token)

        self.tag_dict = Dictionary()
        for token in ["0", "1", "2", "3"]:
            self.tag_dict.add_symbol(token)

        self.item = torch.tensor(
            [
                self.seq_dict.bos(),
                self.seq_dict.index("A"),
                self.seq_dict.index("C"),
                self.seq_dict.index("D"),
                self.seq_dict.index("E"),
                self.seq_dict.eos(),
            ]
        )
        self.tag = torch.tensor(
            [
                self.tag_dict.bos(),
                self.tag_dict.index("1"),
                self.tag_dict.index("2"),
                self.tag_dict.index("3"),
                self.tag_dict.index("0"),
                self.tag_dict.eos(),
            ]
        )
        self.cdr_positions = [1, 2, 3]
        self.mask_idx = self.seq_dict.index("<mask>")

    def _dataset(self, return_masked_tokens, **kwargs):
        return AntibodyMaskTokensDataset(
            [self.item],
            [self.tag],
            self.seq_dict,
            self.tag_dict,
            self.seq_dict.pad(),
            self.mask_idx,
            return_masked_tokens=return_masked_tokens,
            seed=7,
            mask_prob=0.5,
            leave_unmasked_prob=0.0,
            random_token_prob=0.0,
            **kwargs,
        )

    def test_dihedrals_are_invariant_to_batch_partitioning(self):
        generator = torch.Generator().manual_seed(17)
        coordinates = torch.randn(3, 7, 4, 3, generator=generator)
        features = ProteinFeatures()

        batched = features._dihedrals(coordinates)
        per_example = torch.cat(
            [features._dihedrals(coordinates[i:i + 1]) for i in range(3)], dim=0
        )

        self.assertTrue(torch.allclose(batched, per_example, atol=1e-6, rtol=1e-6))

    def test_proteinmpnn_linear_init_matches_lmpro_coordinates(self):
        generator = torch.Generator().manual_seed(23)
        coordinates = torch.randn(9, 14, 3, generator=generator)
        cdr_indices = [3, 4, 5]

        observed = linearize_cdr_coordinates(coordinates, cdr_indices)
        left = coordinates[2]
        right = coordinates[6]
        expected_cdr = torch.stack(
            [left + (right - left) * fraction for fraction in (0.25, 0.5, 0.75)]
        )
        expected_cdr[:, 4:] = expected_cdr[:, 1].unsqueeze(1)

        self.assertTrue(torch.allclose(observed[cdr_indices], expected_cdr))
        self.assertTrue(torch.equal(observed[:3], coordinates[:3]))
        self.assertTrue(torch.equal(observed[6:], coordinates[6:]))

    def test_proteinmpnn_metric_shards_merge_by_counts(self):
        def payload(records, selected, matched, total, score):
            return {
                "cdr_backbone_condition": "linear_init",
                "results": [
                    {
                        "topk": 1,
                        "records": records,
                        "candidates_per_record": 10000,
                        "selected_designs": selected,
                        "matched_positions": matched,
                        "total_positions": total,
                        "aar": matched / total,
                        "mean_proteinmpnn_score": score,
                    }
                ],
            }

        merged = merge_payloads(
            [payload(2, 2, 2, 10, 1.0), payload(1, 1, 9, 10, 3.0)],
            ["shard_00.json", "shard_01.json"],
        )["results"][0]
        self.assertEqual(merged["records"], 3)
        self.assertEqual(merged["matched_positions"], 11)
        self.assertEqual(merged["total_positions"], 20)
        self.assertAlmostEqual(merged["aar"], 0.55)
        self.assertAlmostEqual(merged["mean_proteinmpnn_score"], 5.0 / 3.0)

    def test_proteinmpnn_per_record_shards_are_validated_and_sorted(self):
        def payload(records, matched, total):
            return {
                "cdr_backbone_condition": "linear_init",
                "results": [
                    {
                        "topk": 1,
                        "records": records,
                        "matched_positions": matched,
                        "total_positions": total,
                    }
                ],
            }

        def row(source_line, matched, total):
            return {
                "source_line": str(source_line),
                "name": f"line{source_line:06d}_test",
                "pdb": "test",
                "cdr_backbone_condition": "linear_init",
                "topk": "1",
                "matched_positions": str(matched),
                "total_positions": str(total),
                "aar": str(matched / total),
                "mean_proteinmpnn_score": "2.0",
            }

        merged = merge_per_record_rows(
            [[row(4, 2, 5)], [row(1, 3, 5)]],
            [payload(1, 2, 5), payload(1, 3, 5)],
            "linear_init",
        )
        self.assertEqual([int(item["source_line"]) for item in merged], [1, 4])

        with self.assertRaisesRegex(ValueError, "duplicate per-record key"):
            merge_per_record_rows(
                [[row(1, 2, 5)], [row(1, 3, 5)]],
                [payload(1, 2, 5), payload(1, 3, 5)],
                "linear_init",
            )

    def test_paired_aar_bootstrap_uses_shared_record_resamples(self):
        lmpro = [
            {"source_line": 1, "pdb": "one", "matched": 4, "total": 10},
            {"source_line": 2, "pdb": "two", "matched": 6, "total": 10},
        ]
        proteinmpnn = [
            {
                "source_line": 1,
                "pdb": "one",
                "condition": "linear_init",
                "matched": 2,
                "total": 10,
            },
            {
                "source_line": 2,
                "pdb": "two",
                "condition": "linear_init",
                "matched": 4,
                "total": 10,
            },
        ]
        result = bootstrap_comparison(lmpro, proteinmpnn, resamples=100, seed=7)
        self.assertEqual(result["resampling_unit"], "pdb")
        self.assertEqual(result["clusters"], 2)
        self.assertAlmostEqual(result["lmpro"]["point_estimate"], 0.5)
        self.assertAlmostEqual(result["proteinmpnn"]["point_estimate"], 0.3)
        self.assertAlmostEqual(
            result["delta_lmpro_minus_proteinmpnn"]["point_estimate"], 0.2
        )
        self.assertEqual(
            result["delta_lmpro_minus_proteinmpnn"][
                "bootstrap_probability_above_zero"
            ],
            1.0,
        )

    def test_paired_aar_bootstrap_clusters_repeated_pdb_records(self):
        lmpro = [
            {"source_line": 1, "pdb": "same", "matched": 4, "total": 10},
            {"source_line": 2, "pdb": "same", "matched": 6, "total": 10},
            {"source_line": 3, "pdb": "other", "matched": 8, "total": 10},
        ]
        proteinmpnn = [
            {**row, "condition": "linear_init", "matched": row["matched"] - 1}
            for row in lmpro
        ]
        clustered = bootstrap_comparison(
            lmpro, proteinmpnn, resamples=100, seed=11
        )
        record_level = bootstrap_comparison(
            lmpro, proteinmpnn, resamples=100, seed=11,
            resampling_unit="record",
        )
        self.assertEqual(clustered["records"], 3)
        self.assertEqual(clustered["clusters"], 2)
        self.assertEqual(record_level["clusters"], 3)

    def test_paired_aar_bootstrap_rejects_source_record_mismatch(self):
        lmpro = [
            {"source_line": 1, "pdb": "same", "matched": 4, "total": 10}
        ]
        proteinmpnn = [
            {
                "source_line": 2,
                "pdb": "same",
                "condition": "linear_init",
                "matched": 3,
                "total": 10,
            }
        ]
        with self.assertRaisesRegex(ValueError, "source record differs"):
            bootstrap_comparison(lmpro, proteinmpnn, resamples=10, seed=3)

    def test_lmpro_legacy_tsv_backfill_adds_stable_source_identity(self):
        rows = [
            {"pdb": "one", "topk": "1", "succ": "1", "tot": "4", "rmsd": "2.0"},
            {"pdb": "one", "topk": "10", "succ": "8", "tot": "40", "rmsd": "2.0"},
            {"pdb": "two", "topk": "1", "succ": "2", "tot": "5", "rmsd": "3.0"},
            {"pdb": "two", "topk": "10", "succ": "19", "tot": "50", "rmsd": "3.0"},
        ]
        converted = backfill_rows(rows, [1, 4], ["unused", "one", "x", "x", "two"])
        self.assertEqual([row["source_line"] for row in converted], ["1", "1", "4", "4"])
        self.assertTrue(all("rmsd" not in row for row in converted))
        self.assertEqual(converted[0]["local_cdr3_ca_rmsd"], "2.0")

    def test_structure_metrics_separate_local_and_framework_anchored_rmsd(self):
        native = torch.tensor(
            [[
                [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.4, 0.0, 0.1]],
                [[1.0, 0.0, 0.0], [1.2, 0.3, 0.0], [1.4, 0.0, 0.2]],
                [[0.0, 1.0, 0.0], [0.2, 1.1, 0.3], [0.4, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.3, 0.1, 1.2], [0.4, 0.0, 1.0]],
            ]]
        )
        predicted = native + torch.tensor([2.0, -3.0, 4.0])
        metrics = compute_structure_metrics(
            predicted, native, torch.ones(1, 4, dtype=torch.bool)
        )
        self.assertLess(metrics["local_cdr3_ca_rmsd"].item(), 1e-5)
        self.assertAlmostEqual(
            metrics["framework_anchored_cdr3_backbone_rmsd"].item(),
            (2.0 ** 2 + 3.0 ** 2 + 4.0 ** 2) ** 0.5,
            places=5,
        )

    def test_structure_summary_rejects_duplicate_source_records(self):
        row = {
            "source_line": "1",
            "local_cdr3_ca_rmsd": "1.0",
            "framework_anchored_cdr3_ca_rmsd": "2.0",
            "framework_anchored_cdr3_backbone_rmsd": "3.0",
        }
        with self.assertRaisesRegex(ValueError, "duplicate source_line"):
            summarize_rows([row, row.copy()])

    def test_legacy_masking_keeps_historical_padding(self):
        source = self._dataset(False)[0]
        target = self._dataset(True)[0]
        for position in self.cdr_positions:
            if target[position].item() == self.seq_dict.pad():
                self.assertEqual(source[position].item(), self.seq_dict.pad())
            else:
                self.assertEqual(source[position].item(), self.mask_idx)

    def test_contextual_masking_preserves_unselected_cdr_tokens(self):
        source = self._dataset(False, mask_context_mode="keep_unmasked")[0]
        target = self._dataset(True, mask_context_mode="keep_unmasked")[0]
        for position in self.cdr_positions:
            if target[position].item() == self.seq_dict.pad():
                self.assertEqual(source[position].item(), self.item[position].item())
            else:
                self.assertEqual(source[position].item(), self.mask_idx)

    def test_full_cdr3_span_matches_downstream_context(self):
        source = self._dataset(
            False,
            mask_context_mode="keep_unmasked",
            full_cdr3_mask_prob=1.0,
        )[0]
        target = self._dataset(
            True,
            mask_context_mode="keep_unmasked",
            full_cdr3_mask_prob=1.0,
        )[0]
        self.assertEqual(source[1].item(), self.item[1].item())
        self.assertEqual(source[2].item(), self.item[2].item())
        self.assertEqual(source[3].item(), self.mask_idx)
        self.assertEqual(target[1].item(), self.seq_dict.pad())
        self.assertEqual(target[2].item(), self.seq_dict.pad())
        self.assertEqual(target[3].item(), self.item[3].item())

    def test_fixed_masking_is_stable_across_validation_epochs(self):
        dataset = self._dataset(
            True,
            mask_context_mode="keep_unmasked",
            full_cdr3_mask_prob=0.5,
            fixed_masking=True,
        )
        dataset.set_epoch(1)
        epoch_one = dataset[0].clone()
        dataset.set_epoch(9)
        epoch_nine = dataset[0].clone()
        self.assertTrue(torch.equal(epoch_one, epoch_nine))

    def test_training_masks_refresh_across_epochs(self):
        source, target, mode = AntibodyMaskTokensDataset.apply_mask(
            [self.item],
            [self.tag],
            self.seq_dict,
            self.tag_dict,
            pad_idx=self.seq_dict.pad(),
            mask_idx=self.mask_idx,
            seed=7,
            mask_prob=0.5,
            mask_context_mode="keep_unmasked",
            leave_unmasked_prob=0.0,
            random_token_prob=0.0,
            return_mask_mode=True,
        )
        target_patterns = set()
        for epoch in range(1, 12):
            source.set_epoch(epoch)
            target.set_epoch(epoch)
            mode.set_epoch(epoch)
            target_patterns.add(tuple(target[0].tolist()))
            selected = target[0].ne(self.seq_dict.pad())
            self.assertTrue(torch.equal(source[0].eq(self.mask_idx), selected))
            self.assertEqual(mode[0], 0)
        self.assertGreater(len(target_patterns), 1)

    def test_full_cdr3_mode_metadata_matches_mask(self):
        source, target, mode = AntibodyMaskTokensDataset.apply_mask(
            [self.item],
            [self.tag],
            self.seq_dict,
            self.tag_dict,
            pad_idx=self.seq_dict.pad(),
            mask_idx=self.mask_idx,
            seed=7,
            mask_prob=0.5,
            full_cdr3_mask_prob=1.0,
            mask_context_mode="keep_unmasked",
            leave_unmasked_prob=0.0,
            random_token_prob=0.0,
            return_mask_mode=True,
        )
        self.assertEqual(mode[0], 1)
        self.assertEqual(source[0][3].item(), self.mask_idx)
        self.assertEqual(target[0][3].item(), self.item[3].item())

    def test_rope_runs_and_respects_right_padding(self):
        torch.manual_seed(4)
        attention = MultiheadAttention(
            64,
            4,
            self_attention=True,
            dropout=0.0,
            use_rope=True,
            max_position=32,
        ).eval()
        tokens = torch.randn(8, 2, 64)
        padded_tokens = torch.cat([tokens, torch.randn(4, 2, 64)], dim=0)
        padding_mask = torch.zeros(2, 12, dtype=torch.bool)
        padding_mask[:, 8:] = True

        with torch.no_grad():
            rope_output, _ = attention(tokens, tokens, tokens, need_weights=False)
            padded_output, _ = attention(
                padded_tokens,
                padded_tokens,
                padded_tokens,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            attention.use_rope = False
            no_rope_output, _ = attention(
                tokens, tokens, tokens, need_weights=False
            )

        self.assertTrue(torch.allclose(rope_output, padded_output[:8], atol=1e-6))
        self.assertGreater(
            (rope_output - no_rope_output).abs().max().item(), 1e-6
        )

    def test_architecture_ablation_flags_parse_explicit_zero(self):
        parser = argparse.ArgumentParser()
        AntibodyRobertaModel.add_args(parser)
        args = parser.parse_args(
            ["--use-rope", "0", "--use-residual-mlp", "0"]
        )
        self.assertFalse(args.use_rope)
        self.assertFalse(args.use_residual_mlp)

    def test_token_mean_loss_is_microbatch_partition_invariant(self):
        # Each tuple is (encoder sum, decoder sequence mean, structure mean,
        # token count). The trainer divides the returned sums by total tokens.
        parts = [
            (torch.tensor(12.0), torch.tensor(1.5), torch.tensor(4.0), torch.tensor(4.0)),
            (torch.tensor(18.0), torch.tensor(2.5), torch.tensor(5.0), torch.tensor(6.0)),
        ]
        micro_losses = []
        for values in parts:
            loss, _ = combine_generation_loss(
                *values,
                encoder_scale=1.0,
                decoder_sequence_scale=1.0,
                decoder_structure_scale=0.02,
                mode="token_mean",
            )
            micro_losses.append(loss)

        full_loss, _ = combine_generation_loss(
            torch.tensor(30.0),
            torch.tensor(2.1),
            torch.tensor(4.6),
            torch.tensor(10.0),
            encoder_scale=1.0,
            decoder_sequence_scale=1.0,
            decoder_structure_scale=0.02,
            mode="token_mean",
        )
        self.assertAlmostEqual(
            (sum(micro_losses) / 10.0).item(),
            (full_loss / 10.0).item(),
            places=6,
        )

    def test_geometry_loss_is_microbatch_partition_invariant(self):
        values = torch.tensor(
            [[2.0, 4.0, 0.0], [9.0, 0.0, 0.0], [3.0, 6.0, 12.0]]
        )
        mask = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )
        token_counts = torch.tensor([2.0, 1.0, 3.0])
        full_structure = token_weighted_structure_mean(
            [(values, mask)], token_counts, refine_iterations=2
        )
        full_loss, _ = combine_generation_loss(
            torch.tensor(0.0),
            torch.tensor(0.0),
            full_structure,
            token_counts.sum(),
            encoder_scale=0.0,
            decoder_sequence_scale=0.0,
            decoder_structure_scale=1.0,
            mode="token_mean",
        )

        partitioned_losses = []
        for indices in (slice(0, 2), slice(2, 3)):
            part_tokens = token_counts[indices]
            part_structure = token_weighted_structure_mean(
                [(values[indices], mask[indices])],
                part_tokens,
                refine_iterations=2,
            )
            part_loss, _ = combine_generation_loss(
                torch.tensor(0.0),
                torch.tensor(0.0),
                part_structure,
                part_tokens.sum(),
                encoder_scale=0.0,
                decoder_sequence_scale=0.0,
                decoder_structure_scale=1.0,
                mode="token_mean",
            )
            partitioned_losses.append(part_loss)
        self.assertAlmostEqual(
            sum(partitioned_losses).item(), full_loss.item(), places=6
        )

    def test_full_distance_mask_removes_same_residue_pairs(self):
        coordinates = torch.randn(1, 3, 4, 3)
        atom_mask = torch.ones(1, 3, 4)
        _, pair_mask = full_square_dist(
            coordinates,
            coordinates,
            atom_mask,
            atom_mask,
            remove_diag=True,
        )
        for residue in range(3):
            self.assertEqual(pair_mask[0, residue, residue].sum().item(), 0.0)
        self.assertGreater(pair_mask[0, 0, 1].sum().item(), 0.0)

    def test_kabsch_rejects_reflections(self):
        points = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        )
        mirrored = points.clone()
        mirrored[..., 0] *= -1
        mask = torch.ones(1, 4)
        _, rotation, _ = kabsch(points, mirrored, mask)
        rmsd = compute_rmsd(points, mirrored, mask)
        self.assertGreater(torch.det(rotation).item(), 0.0)
        self.assertGreater(rmsd.item(), 0.1)

    def test_masked_kabsch_ignores_right_padding(self):
        points = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        )
        shifted = points + torch.tensor([2.0, -3.0, 4.0])
        base_rmsd = compute_rmsd(points, shifted, torch.ones(1, 4))
        padded_points = torch.cat([points, torch.tensor([[[99.0, 20.0, -5.0]]])], dim=1)
        padded_shifted = torch.cat([shifted, torch.tensor([[[-40.0, 8.0, 70.0]]])], dim=1)
        padded_rmsd = compute_rmsd(
            padded_points,
            padded_shifted,
            torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]]),
        )
        self.assertTrue(torch.allclose(base_rmsd, padded_rmsd, atol=1e-6))

    def test_mlm_criterion_reports_mask_modes(self):
        class Task:
            target_dictionary = self.seq_dict

        class Model:
            def __init__(self, logits):
                self.logits = logits

            def __call__(self, **unused):
                return (self.logits,)

            @staticmethod
            def get_targets(sample, unused):
                return sample["target"]

        target = torch.full((2, 3), self.seq_dict.pad(), dtype=torch.long)
        target[0, 1] = self.seq_dict.index("A")
        target[1, 1] = self.seq_dict.index("C")
        logits = torch.zeros(2, len(self.seq_dict))
        logits[0, self.seq_dict.index("A")] = 5.0
        logits[1, self.seq_dict.index("D")] = 5.0
        logits.requires_grad_()
        sample = {
            "target": target,
            "mask_mode": torch.tensor([1, 0]),
            "net_input0": {"src_tokens": target},
            "net_input1": {"src_tokens": target},
            "ntokens": 6,
            "nsentences": 2,
        }
        criterion = AntibodyMaskedLmLoss(Task())
        loss, sample_size, logging = criterion(Model(logits), sample)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(logging["full_cdr3_loss"].requires_grad)
        self.assertFalse(logging["partial_loss"].requires_grad)
        self.assertEqual(sample_size.item(), 2)
        self.assertEqual(logging["full_cdr3_sample_size"].item(), 1)
        self.assertEqual(logging["full_cdr3_ncorrect"].item(), 1)
        self.assertEqual(logging["full_cdr3_nexact"].item(), 1)
        self.assertEqual(logging["partial_sample_size"].item(), 1)
        self.assertEqual(logging["partial_ncorrect"].item(), 0)

    def test_refinement_structure_term_only_has_legacy_token_division(self):
        sequence_sum = torch.tensor(100.0)
        structure_sum = torch.tensor(250.0)
        token_mean_seq, token_mean_structure = normalize_refinement_losses(
            sequence_sum, structure_sum, torch.tensor(10.0), 5, "token_mean"
        )
        legacy_seq, legacy_structure = normalize_refinement_losses(
            sequence_sum, structure_sum, torch.tensor(10.0), 5, "legacy"
        )
        self.assertEqual(token_mean_seq.item(), legacy_seq.item())
        self.assertEqual(token_mean_structure.item(), legacy_structure.item() * 10.0)

    def test_stack_legacy_path_keeps_its_historical_refinement_sum(self):
        sequence_sum = torch.tensor(100.0)
        structure_sum = torch.tensor(250.0)
        legacy_seq, legacy_structure = normalize_refinement_losses(
            sequence_sum,
            structure_sum,
            torch.tensor(10.0),
            5,
            "legacy",
            legacy_average_refinement=False,
        )
        corrected_seq, corrected_structure = normalize_refinement_losses(
            sequence_sum,
            structure_sum,
            torch.tensor(10.0),
            5,
            "token_mean",
        )
        self.assertEqual(legacy_seq.item(), corrected_seq.item() * 5.0)
        self.assertEqual(legacy_structure.item(), corrected_structure.item() / 2.0)


if __name__ == "__main__":
    unittest.main()
