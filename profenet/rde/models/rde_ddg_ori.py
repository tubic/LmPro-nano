# rde/models/rde_ddg_ori.py (最终修订版 - 使用Forward Hook)

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from rde.modules.encoders.single import PerResidueEncoder
from rde.modules.encoders.pair import ResiduePairEncoder
from rde.modules.encoders.attn import GAEncoder
from rde.utils.protein.constants import BBHeavyAtom
from rde.models.enhanced_rde import EnhancedRDE

logger = logging.getLogger(__name__)


class DDG_RDE_Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        logger.info("正在加载并完全冻结预训练的 EnhancedRDE 骨干网络...")
        ckpt_path = cfg.model.rde_checkpoint
        state = torch.load(ckpt_path, map_location="cpu")
        pretrain_config = state.get("config", {})
        
        self.rde = EnhancedRDE(pretrain_config)
        self.rde.load_state_dict(state.get("model_state_dict", state), strict=False)
        
        for p in self.rde.parameters():
            p.requires_grad_(False)
        
        dim = self.rde.hidden_dim

        # ## MODIFICATION START: 使用Forward Hook来捕获内部特征 ##
        self.extracted_features = None  # 创建一个用于存储特征的占位符

        def hook_fn(module, input, output):
            """这个函数将在self.rde.output_norm层前向传播后被自动调用"""
            self.extracted_features = output

        # 将hook注册到预训练模型的output_norm层上
        self.rde.output_norm.register_forward_hook(hook_fn)
        # ## MODIFICATION END ##

        logger.info("正在构建任务特异性的可训练编码器通道...")
        self.single_encoder = PerResidueEncoder(
            feat_dim=cfg.encoder.node_feat_dim,
            max_num_atoms=5,
        )
        self.single_fusion = nn.Sequential(
            nn.Linear(cfg.encoder.node_feat_dim + dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.mut_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0)
        self.pair_encoder = ResiduePairEncoder(
            feat_dim=cfg.encoder.pair_feat_dim,
            max_num_atoms=15,
        )
        self.attn_encoder = GAEncoder(**vars(cfg.encoder))
        self.ddg_readout = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def _encode_rde(self, batch):
        # ## MODIFICATION START: 修改特征提取逻辑 ##
        batch_dict = {
            "aa": batch["aa"], 
            "mask": batch["mask_atoms"][:, :, BBHeavyAtom.CA], 
            "pos_heavyatom": batch["pos_atoms"]
        }
        
        # 正常运行预训练模型的前向传播。
        # 我们不需要它的返回值，因为hook会自动捕获我们需要的特征。
        self.rde(batch_dict)
        
        # 从占位符中返回被hook捕获的特征
        return self.extracted_features
        # ## MODIFICATION END ##

    def encode(self, batch):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]

        x_single = self.single_encoder(
            aa=batch['aa'],
            phi=batch['phi'], phi_mask=batch['phi_mask'],
            psi=batch['psi'], psi_mask=batch['psi_mask'],
            chi=chi, chi_mask=batch['chi_mask'],
            mask_residue=mask_residue,
        )
        
        x_pret = self._encode_rde(batch)
        
        x = self.single_fusion(torch.cat([x_single, x_pret], dim=-1))
        b = self.mut_bias(batch['mut_flag'].long())
        x = x + b

        z = self.pair_encoder(
            aa=batch['aa'],
            res_nb=batch['res_nb'], chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'],
        )
        x = self.attn_encoder(
            pos_atoms=batch['pos_atoms'],
            res_feat=x, pair_feat=z,
            mask=mask_residue
        )
        return x

    def forward(self, batch):
        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}
        batch_mt['aa'] = batch_mt['aa_mut']

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)

        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)
        
        loss = (F.mse_loss(ddg_pred, batch['ddG']) + F.mse_loss(ddg_pred_inv, -batch['ddG'])) / 2
        
        loss_dict = {'regression': loss}
        out_dict = {'ddG_pred': ddg_pred, 'ddG_true': batch['ddG']}
        return loss_dict, out_dict
