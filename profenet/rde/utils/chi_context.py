"""Shared contracts for side-chain targets and corrupted-angle hints."""

from __future__ import annotations

import torch


CHI_TARGET_MODES = ("all", "corrupt", "masked")


def build_chi_target_mask(batch, mode="all"):
    """Return the valid chi entries supervised by the requested objective."""
    if mode not in CHI_TARGET_MODES:
        raise ValueError(
            f"chi target mode must be one of {CHI_TARGET_MODES}, got {mode!r}"
        )
    if "chi_mask" not in batch:
        raise KeyError("chi_mask is required to build chi targets")

    target_mask = batch["chi_mask"].bool()
    residue_mask = batch.get("mask")
    if residue_mask is not None:
        target_mask = target_mask & residue_mask.bool().unsqueeze(-1)

    flag_key = {
        "all": None,
        "corrupt": "chi_corrupt_flag",
        "masked": "chi_masked_flag",
    }[mode]
    if flag_key is not None:
        if flag_key not in batch:
            raise KeyError(f"{flag_key} is required for chi target mode {mode!r}")
        flag = batch[flag_key].bool()
        if flag.shape != target_mask.shape[:-1]:
            raise ValueError(
                f"{flag_key} shape {tuple(flag.shape)} does not match chi targets "
                f"{tuple(target_mask.shape)}"
            )
        target_mask = target_mask & flag.unsqueeze(-1)
    return target_mask


def encode_chi_hint(angles, chi_mask=None, masked_flag=None, use_hint=True):
    """Encode chi angles as flattened sin/cos pairs with an unambiguous mask token.

    A real zero-radian angle maps to ``(0, 1)``. Missing or withheld angles map
    to ``(0, 0)``, which is outside the unit circle and therefore cannot be
    confused with a physical angle.
    """
    if angles.ndim < 2 or angles.shape[-1] != 4:
        raise ValueError(f"chi angles must end in four torsions, got {angles.shape}")

    encoded = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
    available = torch.ones_like(angles, dtype=torch.bool)
    if chi_mask is not None:
        if chi_mask.shape != angles.shape:
            raise ValueError(
                f"chi_mask shape {tuple(chi_mask.shape)} does not match angles "
                f"{tuple(angles.shape)}"
            )
        available = available & chi_mask.bool()
    if masked_flag is not None:
        if masked_flag.shape != angles.shape[:-1]:
            raise ValueError(
                f"masked_flag shape {tuple(masked_flag.shape)} does not match angles "
                f"{tuple(angles.shape)}"
            )
        available = available & ~masked_flag.bool().unsqueeze(-1)
    if not use_hint:
        available = torch.zeros_like(available)

    encoded = torch.where(
        available.unsqueeze(-1), encoded, torch.zeros_like(encoded)
    )
    return encoded.flatten(start_dim=-2)
