# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
from fairseq.data import Dictionary, data_utils
from fairseq.data import BaseWrapperDataset, LRUCacheDataset


class AntibodyMaskTokensDataset(BaseWrapperDataset):
    """Mask CDR residues for AbBERT pretraining.

    ``legacy_pad`` retains the historical behavior where every CDR token not
    selected as a target is replaced by padding. ``keep_unmasked`` implements
    normal masked-LM semantics and keeps those native residues visible. The
    latter is required for a faithful downstream-style CDR3 span objective.
    """

    @classmethod
    def apply_mask(
        cls,
        dataset1: torch.utils.data.Dataset,
        dataset2: torch.utils.data.Dataset,
        *args,
        return_mask_mode: bool = False,
        **kwargs
    ):
        """Return source/target datasets and, optionally, the masking mode.

        Only the immutable token datasets are LRU-cached. Masked views keep
        their epoch in the cache key, so accessing an item in one epoch cannot
        freeze its mask for later epochs.
        """
        dataset1 = LRUCacheDataset(dataset1)
        dataset2 = LRUCacheDataset(dataset2)
        datasets = (
            cls(dataset1, dataset2, *args, **kwargs, return_masked_tokens=False),
            cls(dataset1, dataset2, *args, **kwargs, return_masked_tokens=True),
        )
        if return_mask_mode:
            datasets += (
                cls(dataset1, dataset2, *args, **kwargs, return_mask_mode=True),
            )
        return datasets

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        tag_dataset: torch.utils.data.Dataset,
        seq_vocab: Dictionary,
        tag_vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        return_mask_mode: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        full_cdr3_mask_prob: float = 0.0,
        mask_context_mode: str = "legacy_pad",
        fixed_masking: bool = False,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        mask_multiple_length: int = 1,
        mask_stdev: float = 0.0,
    ):
        assert not (return_masked_tokens and return_mask_mode)
        assert 0.0 < mask_prob <= 1.0
        assert 0.0 <= full_cdr3_mask_prob <= 1.0
        assert mask_context_mode in ("legacy_pad", "keep_unmasked")
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0
        assert mask_multiple_length >= 1
        assert mask_stdev >= 0.0

        self.dataset = dataset
        self.tag_dataset = tag_dataset
        self.seq_vocab = seq_vocab
        self.tag_vocab = tag_vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.return_mask_mode = return_mask_mode
        self.seed = seed
        self.mask_prob = mask_prob
        self.full_cdr3_mask_prob = full_cdr3_mask_prob
        self.mask_context_mode = mask_context_mode
        self.fixed_masking = fixed_masking
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.mask_multiple_length = mask_multiple_length
        self.mask_stdev = mask_stdev

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(self.seq_vocab.count)
            else:
                weights = np.ones(len(self.seq_vocab))
            weights[: self.seq_vocab.nspecial] = 0
            self.weights = weights / weights.sum()

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        # Training sees a fresh deterministic mask each epoch. Validation and
        # test keep epoch zero so checkpoint losses always score identical targets.
        self.epoch = 0 if self.fixed_masking else epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    def collater(self, samples):
        if self.return_mask_mode:
            return torch.tensor(samples, dtype=torch.long)
        return super().collater(samples)

    def _draw_target_mask(self, tag, cdr_mask, cdr3_idx, size):
        if (
            self.full_cdr3_mask_prob > 0.0
            and np.random.rand() < self.full_cdr3_mask_prob
        ):
            mask = (tag[cdr_mask] == cdr3_idx).cpu().numpy().astype(bool)
            return mask, True
        if self.mask_prob == 1.0:
            return np.full(size, True), False

        mask = np.full(size, False)
        num_mask = int(self.mask_prob * size + np.random.rand())
        mask_idc = np.random.choice(size, num_mask, replace=False)
        mask[mask_idc] = True
        return mask, False

    def _draw_replacements(self, mask):
        rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
        if rand_or_unmask_prob == 0.0:
            return mask, None

        rand_or_unmask = mask & (
            np.random.rand(mask.size) < rand_or_unmask_prob
        )
        if self.random_token_prob == 0.0:
            return mask ^ rand_or_unmask, None
        if self.leave_unmasked_prob == 0.0:
            return mask, rand_or_unmask

        unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
        unmask = rand_or_unmask & (np.random.rand(mask.size) < unmask_prob)
        return mask ^ unmask, rand_or_unmask & (~unmask)

    @lru_cache(maxsize=8)
    def __getitem_cached__(self, seed: int, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            tag = self.tag_dataset[index]
            cdr1_idx = self.tag_vocab.index("1")
            cdr2_idx = self.tag_vocab.index("2")
            cdr3_idx = self.tag_vocab.index("3")
            cdr_mask = torch.logical_or(
                torch.logical_or(tag == cdr1_idx, tag == cdr2_idx),
                tag == cdr3_idx,
            )

            assert self.mask_idx not in item, (
                "Dataset contains mask_idx (={}), this is not expected!".format(
                    self.mask_idx
                )
            )

            cdr_positions = cdr_mask.cpu().numpy().astype(bool)
            item_np = item.cpu().numpy()
            cdr_values = item_np[cdr_positions]
            mask, is_full_cdr3 = self._draw_target_mask(
                tag, cdr_mask, cdr3_idx, len(cdr_values)
            )

            if self.return_mask_mode:
                return int(is_full_cdr3)

            if self.return_masked_tokens:
                new_item = np.full(len(item_np), self.pad_idx, dtype=item_np.dtype)
                targets = np.full(len(cdr_values), self.pad_idx, dtype=item_np.dtype)
                targets[mask] = cdr_values[mask]
                new_item[cdr_positions] = targets
                return torch.from_numpy(new_item)

            mask, rand_mask = self._draw_replacements(mask)
            new_item = np.array(item_np, copy=True)
            if self.mask_context_mode == "keep_unmasked":
                masked_cdr = np.array(cdr_values, copy=True)
            else:
                # Reproduce the old, non-standard pretraining corpus exactly.
                masked_cdr = np.full(
                    len(cdr_values), self.pad_idx, dtype=item_np.dtype
                )

            masked_cdr[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    masked_cdr[rand_mask] = np.random.choice(
                        len(self.seq_vocab), num_rand, p=self.weights
                    )
            new_item[cdr_positions] = masked_cdr
            return torch.from_numpy(new_item)
