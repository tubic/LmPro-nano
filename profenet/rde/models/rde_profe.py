"""ProFE-style antisymmetric ddG head on the published RDE backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rde.models.rde import CircularSplineRotamerDensityEstimator
from rde.modules.encoders.attn import GAEncoder
from rde.modules.encoders.pair import ResiduePairEncoder
from rde.modules.encoders.single import PerResidueEncoder
from rde.utils.protein.constants import BBHeavyAtom


class ProFERDENetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model

        checkpoint = torch.load(
            model_cfg.rde_checkpoint, map_location="cpu", weights_only=False
        )
        if "config" in checkpoint and "model" in checkpoint:
            backbone_cfg = checkpoint["config"].model
            backbone_state = checkpoint["model"]
        else:
            # Release artifacts are bare state_dicts.  Their matching
            # architecture is provided explicitly by the inference wrapper.
            backbone_cfg = getattr(model_cfg, "backbone_config", None)
            if backbone_cfg is None:
                raise ValueError(
                    "a bare RDE state_dict requires model.backbone_config"
                )
            backbone_state = checkpoint
        self.rde = CircularSplineRotamerDensityEstimator(backbone_cfg)
        self.rde.load_state_dict(backbone_state, strict=True)
        self.rde.eval()
        for parameter in self.rde.parameters():
            parameter.requires_grad_(False)

        backbone_dim = int(backbone_cfg.encoder.node_feat_dim)
        expected_dim = int(getattr(model_cfg, "backbone_dim", backbone_dim))
        if expected_dim != backbone_dim:
            raise ValueError(
                "backbone_dim=%d does not match RDE checkpoint dimension=%d"
                % (expected_dim, backbone_dim)
            )
        encoder_cfg = model_cfg.encoder
        if int(encoder_cfg.node_feat_dim) != backbone_dim:
            raise ValueError("The trainable encoder and RDE backbone dimensions must match")

        self.single_encoder = PerResidueEncoder(
            feat_dim=encoder_cfg.node_feat_dim,
            max_num_atoms=5,
        )
        self.single_fusion = nn.Sequential(
            nn.Linear(2 * backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, backbone_dim),
        )
        self.mut_bias = nn.Embedding(2, backbone_dim, padding_idx=0)
        self.pair_encoder = ResiduePairEncoder(
            feat_dim=encoder_cfg.pair_feat_dim,
            max_num_atoms=5,
        )
        encoder_options = (
            dict(encoder_cfg)
            if isinstance(encoder_cfg, dict)
            else dict(vars(encoder_cfg))
        )
        self.attn_encoder = GAEncoder(**encoder_options)

        self.ddg_readout = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, 1),
        )
        self.pooling = str(getattr(model_cfg, "pooling", "max"))
        if self.pooling not in {"max", "mutation_mean"}:
            raise ValueError("model.pooling must be max or mutation_mean")
        self.antisymmetric_output = bool(
            getattr(model_cfg, "antisymmetric_output", True)
        )

    def train(self, mode=True):
        super().train(mode)
        self.rde.eval()
        return self

    @staticmethod
    def _atom_view(batch):
        positions = (
            batch["pos_atoms"]
            if "pos_atoms" in batch
            else batch["pos_heavyatom"][:, :, :5]
        )
        atom_mask = (
            batch["mask_atoms"]
            if "mask_atoms" in batch
            else batch["mask_heavyatom"][:, :, :5]
        )
        return positions, atom_mask

    def _get_rde_context(self, batch):
        positions, atom_mask = self._atom_view(batch)
        rde_batch = dict(batch)
        rde_batch["pos_atoms"] = positions
        rde_batch["mask_atoms"] = atom_mask
        rde_batch["chi_corrupt"] = batch["chi"]
        rde_batch["chi_masked_flag"] = batch["mut_flag"]
        with torch.no_grad():
            return self.rde.encode(rde_batch)

    def encode(self, batch):
        positions, atom_mask = self._atom_view(batch)
        residue_mask = atom_mask[:, :, BBHeavyAtom.CA]
        chi = batch["chi"] * (~batch["mut_flag"].bool())[:, :, None]
        single = self.single_encoder(
            aa=batch["aa"],
            phi=batch["phi"],
            phi_mask=batch["phi_mask"],
            psi=batch["psi"],
            psi_mask=batch["psi_mask"],
            chi=chi,
            chi_mask=batch["chi_mask"],
            mask_residue=residue_mask,
        )
        context = batch.get("x_pret")
        if context is None:
            context = self._get_rde_context(batch)
        context = context.to(single.dtype)
        features = self.single_fusion(torch.cat([single, context], dim=-1))
        features = features + self.mut_bias(batch["mut_flag"].long())

        pair = self.pair_encoder(
            aa=batch["aa"],
            res_nb=batch["res_nb"],
            chain_nb=batch["chain_nb"],
            pos_atoms=positions,
            mask_atoms=atom_mask,
        )
        return self.attn_encoder(
            pos_atoms=positions,
            res_feat=features,
            pair_feat=pair,
            mask=residue_mask,
        )

    def _pool(self, features, batch):
        if self.pooling == "mutation_mean":
            mask = batch["mut_flag"].bool()
            weights = mask.to(features.dtype)[..., None]
            return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        residue_mask = batch.get(
            "mask", batch["mask_heavyatom"][:, :, BBHeavyAtom.CA]
        ).bool()
        masked = features.masked_fill(~residue_mask[..., None], -torch.inf)
        pooled = masked.max(dim=1).values
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

    def forward(self, batch):
        aa_wt = batch["aa_wt"] if "aa_wt" in batch else batch["aa"]
        wt_batch = {**batch, "aa": aa_wt, "x_pret": batch.get("x_pret_wt")}
        mt_batch = {**batch, "aa": batch["aa_mut"], "x_pret": batch.get("x_pret_mt")}
        wt = self._pool(self.encode(wt_batch), batch)
        mt = self._pool(self.encode(mt_batch), batch)

        forward_prediction = self.ddg_readout(mt - wt).squeeze(-1)
        reverse_prediction = self.ddg_readout(wt - mt).squeeze(-1)
        target = batch["ddG"].view_as(forward_prediction)
        loss = 0.5 * (
            F.mse_loss(forward_prediction, target)
            + F.mse_loss(reverse_prediction, -target)
        )
        if self.antisymmetric_output:
            prediction = 0.5 * (forward_prediction - reverse_prediction)
        else:
            prediction = forward_prediction
        return {"regression": loss, "total_loss": loss}, {"ddg": prediction}


def model_factory(cfg):
    return ProFERDENetwork(cfg)
