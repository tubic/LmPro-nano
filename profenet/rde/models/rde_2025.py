# rde_ddg_new.py (最终架构版)
import os
import logging
from typing import Dict
import types
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

# [核心] 骨干网络依然是我们熟悉的 EnhancedRDE
from rde.models.enhanced_rde import EnhancedRDE
# [移植] 导入新架构所需的、可训练的编码器模块
from rde.modules.encoders.single import PerResidueEncoder
from rde.modules.encoders.pair import ResiduePairEncoder
from rde.modules.encoders.attn import GAEncoder
from rde.utils.protein.constants import AA, BBHeavyAtom

logger = logging.getLogger(__name__)

def ns_to_dict(ns_obj):
    if isinstance(ns_obj, Mapping):
        return {key: ns_to_dict(value) for key, value in ns_obj.items()}
    if isinstance(ns_obj, types.SimpleNamespace):
        return {key: ns_to_dict(value) for key, value in vars(ns_obj).items()}
    if isinstance(ns_obj, (list, tuple)):
        return [ns_to_dict(value) for value in ns_obj]
    return ns_obj


class RDEDeltaDeltaG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. 加载 EnhancedRDE 作为预训练骨干网络，并冻结
        logger.info("Loading EnhancedRDE backbone...")
        ckpt_path = cfg.model.rde_checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # Training checkpoints carry their architecture in ``config``.  The
        # public release stores a bare state_dict to avoid shipping optimizer
        # and runtime state, so its architecture is supplied by the inference
        # wrapper through ``model.backbone_config``.
        release_config = getattr(cfg.model, 'backbone_config', None)
        pretrain_config = ckpt.get(
            'config',
            release_config if release_config is not None else ns_to_dict(cfg),
        )
        pretrain_config = ns_to_dict(pretrain_config)
        self.rde = EnhancedRDE(pretrain_config)
        self.rde.load_state_dict(state_dict, strict=True)
        self.rde.eval()
        for p in self.rde.parameters():
            p.requires_grad_(False)
        
        # 从骨干模型配置中获取特征维度
        dim = cfg.model.backbone_dim
        if int(dim) != int(self.rde.hidden_dim):
            raise ValueError(
                f"backbone_dim={dim} does not match checkpoint hidden_dim={self.rde.hidden_dim}"
            )
        self.backbone_chi_mode = str(
            getattr(cfg.model, 'backbone_chi_mode', 'none')
        )
        if self.backbone_chi_mode not in {'none', 'native', 'mask_mut'}:
            raise ValueError(
                "model.backbone_chi_mode must be one of: none, native, mask_mut"
            )
        logger.info(f"EnhancedRDE chi context mode: {self.backbone_chi_mode}")

        # 2. [移植] 创建一套全新的、可训练的专属编码器
        # [性能] pair_atoms 控制 ResiduePairEncoder 使用的重原子数。
        # pair.py:66 会构造 (B,L,L,A,A,3) 的中间张量，是全流程的主要开销：
        # A=15 时 B=32,L=128 就要 1.4GB；取 A=5（N,CA,C,O,CB，即上游 select_atom
        # 的 backbone+CB 分辨率）可降到 ~157MB，约 9 倍提速。侧链构象信息已由
        # single_encoder 的 chi 角通道提供，故此处不必用全部 15 个原子。
        self.pair_atoms = int(getattr(cfg.model.encoder, 'pair_atoms', 15))
        self.single_encoder = PerResidueEncoder(feat_dim=cfg.model.encoder.node_feat_dim, max_num_atoms=15)
        self.pair_encoder = ResiduePairEncoder(feat_dim=cfg.model.encoder.pair_feat_dim,
                                              max_num_atoms=self.pair_atoms)

        # GAEncoder accepts ordinary dictionaries for its nested options.
        encoder_cfg_dict = ns_to_dict(cfg.model.encoder)
        encoder_cfg_dict.pop('pair_atoms', None)   # 非 GAEncoder 的参数
        self.attn_encoder = GAEncoder(**encoder_cfg_dict)

        # 3. [移植] 创建特征融合层和突变偏置层
        self.single_fusion = nn.Sequential(
            nn.Linear(dim + cfg.model.encoder.node_feat_dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.mut_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0)

        # 4. 预测头。
        # rich=True: 借鉴九月分支 rde_ddg.py 的「多尺度池化 + WT/MT/差值」读出，
        #   每分支取 global(全链) / local(突变残基) / env(突变 10A 邻域, 排除突变残基) 三类池化，
        #   读出输入 = [MT三类, WT三类, 差值三类] = 9*dim。
        #   反对称性由 0.5*(head(mt,wt) - head(wt,mt)) 保证（交换参数再相减）。
        # rich=False: 原先的「仅差值」简单头，保留以便对照。
        head_cfg = getattr(cfg.model, 'head', None)
        self.rich_head = bool(getattr(head_cfg, 'rich', True)) if head_cfg is not None else True
        self.loss_alpha = float(getattr(head_cfg, 'loss_alpha', 0.2)) if head_cfg is not None else 0.2
        self.sign_loss_weight = float(getattr(head_cfg, 'sign_loss_weight', 0.0)) if head_cfg is not None else 0.0
        self.sign_loss_temperature = float(getattr(head_cfg, 'sign_loss_temperature', 1.0)) if head_cfg is not None else 1.0
        self.aggregate_loss_across_accumulation = bool(
            getattr(getattr(cfg, 'training', None), 'aggregate_loss_across_accumulation', False)
        )
        if self.sign_loss_weight < 0.0:
            raise ValueError('head.sign_loss_weight must be non-negative')
        if self.sign_loss_temperature <= 0.0:
            raise ValueError('head.sign_loss_temperature must be positive')
        readout_dim = int(getattr(head_cfg, 'readout_dim', 512)) if head_cfg is not None else 512
        self.env_radius = float(getattr(head_cfg, 'env_radius', 10.0)) if head_cfg is not None else 10.0
        in_dim = dim * 9 if self.rich_head else dim
        if self.rich_head:
            # 九路拼接的异质特征，用 LayerNorm 稳定量纲是合适的
            self.ddg_readout = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, readout_dim), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(readout_dim, 1)
            )
        else:
            # [关键] 差值头绝不能加 LayerNorm：输入是 (H_mt - H_wt)，
            # LayerNorm 会把每个样本的差值向量归一化到单位尺度，抹掉“扰动幅度”，
            # 而 ddG 的大小主要就编码在该差值向量的模长中（归一化后只剩方向）。
            self.ddg_readout = nn.Sequential(
                nn.Linear(in_dim, in_dim), nn.ReLU(),
                nn.Linear(in_dim, 1)
            )
        logger.info(f"ddG head: rich={self.rich_head} in_dim={in_dim} readout_dim={readout_dim} "
                    f"loss_alpha={self.loss_alpha} sign_weight={self.sign_loss_weight} "
                    f"aggregate_loss={self.aggregate_loss_across_accumulation} "
                    f"env_radius={self.env_radius}")

        # 5. [性能] 突变 patch 池化的配置：
        #    patch = 突变残基 ∪ (CA 距任一突变残基 <= radius Å 的残基)，并以 min_k 最近邻兜底。
        #    对该 patch 做 masked-mean，取代原先对整条复合物的 max 池化——
        #    后者会把单点突变的信号稀释到几百个残基里。无新增可训练参数。
        pool_cfg = getattr(cfg.model, 'pool', None)
        self.pool_mode = str(getattr(pool_cfg, 'mode', 'mutation_patch_mean')) if pool_cfg is not None else 'mutation_patch_mean'
        if self.pool_mode not in {'global_max', 'mutation_mean', 'mutation_patch_mean'}:
            raise ValueError(
                "model.pool.mode must be global_max, mutation_mean, or mutation_patch_mean"
            )
        self.pool_radius = float(getattr(pool_cfg, 'radius', 10.0)) if pool_cfg is not None else 10.0
        self.pool_min_k = int(getattr(pool_cfg, 'min_k', 16)) if pool_cfg is not None else 16
        logger.info(
            f"ddG pooling={self.pool_mode} "
            f"(radius={self.pool_radius}A, min_k={self.pool_min_k})"
        )

    def _patch_mask(self, batch) -> torch.Tensor:
        """ 计算突变 patch 的布尔掩码 [B, N]（wt/mt 共用，只依赖 mut_flag 与坐标）。 """
        mut = batch['mut_flag'].bool()                                  # [B,N]
        aa_wt = batch['aa_wt'] if 'aa_wt' in batch else batch['aa']
        resmask = self._residue_mask(batch, aa_wt)
        ca = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA, :]           # [B,N,3]
        B, N, _ = ca.shape
        out = torch.zeros(B, N, dtype=torch.bool, device=ca.device)
        for b in range(B):
            m = mut[b] & resmask[b]
            if int(m.sum()) == 0:
                # 没有标出突变残基时退化为整链有效残基（不至于空集）
                out[b] = resmask[b]
                continue
            d = torch.cdist(ca[b], ca[b][m]).min(dim=1).values         # [N] 到最近突变残基的 CA 距离
            d = d.masked_fill(~resmask[b], float('inf'))
            within = d <= self.pool_radius
            if int(within.sum()) < self.pool_min_k:                    # 邻域太小则用 k 最近兜底
                k = int(min(self.pool_min_k, int(resmask[b].sum())))
                idx = torch.topk(d, k, largest=False).indices
                within = torch.zeros_like(resmask[b])
                within[idx] = True
            out[b] = within | m                                        # 始终包含突变残基本身
        return out

    @staticmethod
    def _masked_mean(h: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        """ 对 patch 内残基特征做均值池化。h:[B,N,D] patch:[B,N] -> [B,D] """
        w = patch.to(h.dtype)[..., None]                               # [B,N,1]
        return (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_max(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = h.masked_fill(~mask.bool()[..., None], -torch.inf).max(dim=1).values
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

    @staticmethod
    def _residue_mask(batch, aa):
        if 'mask' in batch:
            return batch['mask'].bool()
        if 'mask_heavyatom' in batch:
            return batch['mask_heavyatom'][:, :, BBHeavyAtom.CA].bool()
        if 'mask_atoms' in batch:
            return batch['mask_atoms'][:, :, BBHeavyAtom.CA].bool()
        return torch.ones_like(aa, dtype=torch.bool)

    def _scale_masks(self, batch):
        """ 返回三个尺度的布尔掩码 [B,N]: 全链有效残基 / 突变残基 / 突变邻域(排除突变本身)。 """
        mut = batch['mut_flag'].bool()
        aa_wt = batch['aa_wt'] if 'aa_wt' in batch else batch['aa']
        resmask = self._residue_mask(batch, aa_wt)
        ca = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA, :]
        B, N, _ = ca.shape
        env = torch.zeros(B, N, dtype=torch.bool, device=ca.device)
        for b in range(B):
            m = mut[b] & resmask[b]
            if int(m.sum()) == 0:
                continue
            d = torch.cdist(ca[b], ca[b][m]).min(dim=1).values
            d = d.masked_fill(~resmask[b], float('inf'))
            env[b] = (d <= self.env_radius) & (~m) & resmask[b]
        return resmask, (mut & resmask), env

    def _pool_feats(self, h, masks):
        resmask, mut, env = masks
        return torch.cat([self._masked_mean(h, resmask),
                          self._masked_mean(h, mut),
                          self._masked_mean(h, env)], dim=-1)      # [B, 3*D]

    @staticmethod
    def _pearson_loss(pred, true, eps=1e-6):
        """ 1 - Pearson。batch 过小时退化（无梯度），由调用方跳过。 """
        vx = pred.view(-1) - pred.view(-1).mean()
        vy = true.view(-1) - true.view(-1).mean()
        return 1.0 - (vx * vy).sum() / (vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt() + eps)

    def loss_from_branches(self, fwd, rev, target):
        """Compute the ddG loss, optionally over a complete effective batch.

        ``train_accumulated_update`` uses this hook for candidates that opt in
        to cross-microbatch loss aggregation.  Keeping it on the model preserves
        the existing behavior for the locked historical configurations.
        """
        a = self.loss_alpha
        use_corr = (a < 1.0) and (fwd.numel() >= 4)

        def _branch_loss(pred, true):
            value = F.mse_loss(pred, true)
            if use_corr:
                value = a * value + (1.0 - a) * self._pearson_loss(pred, true)
            return value

        loss = 0.5 * (_branch_loss(fwd, target) + _branch_loss(rev, -target))
        if self.sign_loss_weight > 0.0:
            temperature = self.sign_loss_temperature
            antisymmetric_prediction = 0.5 * (fwd - rev)
            sign_labels = (target > 0).to(antisymmetric_prediction.dtype)
            sign_loss = F.binary_cross_entropy_with_logits(
                temperature * antisymmetric_prediction,
                sign_labels,
            )
            loss = loss + self.sign_loss_weight * sign_loss
        return loss

    def train(self, mode: bool = True):
        """ 覆盖 train() 以保证冻结的 EnhancedRDE 骨干始终处于 eval 模式。
        否则 model.train(True) 会递归地把骨干切回训练模式，导致特征提取阶段
        dropout/LayerNorm 行为在训练与评估间不一致（骨干本应作为确定性特征提取器）。"""
        super().train(mode)
        self.rde.eval()
        return self

    def _get_rde_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """ 使用（默认冻结的）EnhancedRDE 提取预训练特征 (上下文)。

        pret_grad=True 时允许梯度穿过骨干（用于部分解冻微调）；否则走 no_grad 以省显存。
        """
        want_grad = bool(getattr(self, 'pret_grad', False)) and self.training
        rde = self.rde
        aa = batch['aa']
        mask = self._residue_mask(batch, aa)
        invalid_valid = mask & ((aa < 0) | (aa >= len(AA)))
        if invalid_valid.any():
            raise ValueError('Valid residues contain amino-acid indices outside EnhancedRDE vocabulary')
        aa_safe = aa.masked_fill(~mask, 0)
        batch_size, num_residues = aa.shape
        pos_heavyatom = batch['pos_heavyatom']
        pos_alpha = pos_heavyatom[:, :, BBHeavyAtom.CA, :]

        # Everything before the optionally unfrozen top fusion layers remains
        # frozen, so retaining its autograd graph would only waste memory.
        with torch.no_grad():
            local_frames = rde._get_backbone_local_frames(pos_heavyatom)
            v_init = torch.zeros(
                batch_size,
                num_residues,
                rde.v_dim,
                3,
                device=aa.device,
                dtype=pos_alpha.dtype,
            )
            num_vectors = min(3, rde.v_dim)
            v_init[:, :, :num_vectors, :] = local_frames[:, :, :num_vectors, :]
            v_init = v_init * mask[..., None, None]

            aa_onehot = F.one_hot(aa_safe, num_classes=len(AA)).float()
            physicochemical = torch.stack(
                [
                    rde.hydrophobicity[aa_safe],
                    rde.mw_weight[aa_safe],
                    rde.charge[aa_safe],
                ],
                dim=-1,
            )
            seq_context = rde.aa_embed(aa_safe)

            if self.backbone_chi_mode == 'none':
                chi_context = torch.zeros(
                    batch_size,
                    num_residues,
                    4,
                    device=aa.device,
                    dtype=seq_context.dtype,
                )
                chi_sincos = rde.prepare_chi_hint(
                    chi_context,
                    chi_mask=batch.get('chi_mask'),
                    use_hint=False,
                    apply_dropout=False,
                )
            else:
                if 'chi' not in batch:
                    raise KeyError(
                        f"chi is required for backbone_chi_mode={self.backbone_chi_mode}"
                    )
                chi_context = batch['chi'].to(seq_context.dtype)
                masked_flag = None
                if self.backbone_chi_mode == 'mask_mut':
                    if 'mut_flag' not in batch:
                        raise KeyError('mut_flag is required for backbone_chi_mode=mask_mut')
                    masked_flag = batch['mut_flag'].bool()
                chi_sincos = rde.prepare_chi_hint(
                    chi_context,
                    chi_mask=batch.get('chi_mask'),
                    masked_flag=masked_flag,
                    use_hint=True,
                    apply_dropout=False,
                )

            rel_pos = pos_alpha[:, :, None, :] - pos_alpha[:, None, :, :]
            rel_dist = torch.norm(rel_pos, p=2, dim=-1)
            node_pair_feat = torch.cat(
                [
                    repeat(seq_context, 'b i d -> b i j d', j=num_residues),
                    repeat(seq_context, 'b j d -> b i j d', i=num_residues),
                    repeat(chi_sincos, 'b j d -> b i j d', i=num_residues),
                ],
                dim=-1,
            )
            dist_emb = rde.dist_proj(rde.dist_rbf(rel_dist))
            pair_mask = (mask[:, :, None] & mask[:, None, :]).to(dist_emb.dtype)
            pair_rep_init = (
                rde.node_pair_embed(node_pair_feat) + dist_emb
            ) * pair_mask[..., None]

            geometry_feat = torch.cat([aa_onehot, physicochemical], dim=-1)
            geometry_config = rde.config.get('model', {}).get('gnn', {})
            edge_index = rde._build_graph(
                pos_alpha,
                mask,
                use_radius=geometry_config.get('use_radius', False),
                radius=geometry_config.get('radius', 12.0),
                k=geometry_config.get('k_neighbors', 16),
            )
            geometry_context = rde.geometry_encoder(
                geometry_feat.view(batch_size * num_residues, -1),
                pos_alpha.reshape(batch_size * num_residues, 3),
                edge_index,
                v_init=v_init.view(batch_size * num_residues, rde.v_dim, 3),
            ).view(batch_size, num_residues, rde.hidden_dim)
            geometry_context = geometry_context * mask[..., None]

        num_layers = len(rde.fusion_stack)
        grad_layers = (
            int(getattr(self, 'pret_grad_layers', 0)) if want_grad else 0
        )
        prev_pair_rep = torch.zeros_like(pair_rep_init)
        prev_seq_context = torch.zeros_like(seq_context)
        for _ in range(rde.num_recycle + 1):
            seq_context_recycled = seq_context + prev_seq_context.detach()
            pair_rep_recycled = pair_rep_init + prev_pair_rep.detach()
            for layer_index, layer in enumerate(rde.fusion_stack):
                enable_grad = grad_layers > 0 and layer_index >= num_layers - grad_layers
                with torch.set_grad_enabled(enable_grad):
                    seq_context_recycled, pair_rep_recycled = layer(
                        seq_context_recycled,
                        geometry_context,
                        pair_rep_recycled,
                        mask,
                    )
            prev_pair_rep = pair_rep_recycled
            prev_seq_context = seq_context_recycled

        with torch.set_grad_enabled(grad_layers > 0):
            return rde.output_norm(seq_context_recycled)

    def encode(self, batch):
        """ [移植] 仿照 rde_ddg.py 的完整编码流程 """
        mask_residue = batch['mask_heavyatom'][:, :, BBHeavyAtom.CA]

        # a. 用可训练的编码器计算新特征 (x_single)
        # 假设数据加载器提供了chi, phi, psi等信息
        x_single = self.single_encoder(
            aa = batch['aa'],
            phi = batch.get('phi', torch.zeros_like(mask_residue).float()), phi_mask = batch.get('phi_mask', torch.zeros_like(mask_residue).bool()),
            psi = batch.get('psi', torch.zeros_like(mask_residue).float()), psi_mask = batch.get('psi_mask', torch.zeros_like(mask_residue).bool()),
            chi = batch.get('chi', torch.zeros_like(batch['chi_mask']).float()), chi_mask = batch['chi_mask'],
            mask_residue = mask_residue,
        )

        # b. 提取预训练特征 (x_pret)
        # [性能] 骨干已冻结且恒为 eval、图构建确定性，故其输出对给定输入是确定的。
        # 若上游提供了预计算的 'x_pret'，直接使用——数学上与现算完全等价，仅为加速。
        if batch.get('x_pret', None) is not None:
            x_pret = batch['x_pret'].to(x_single.dtype)
        else:
            x_pret = self._get_rde_context(batch)

        # c. 融合两套特征
        x = self.single_fusion(torch.cat([x_single, x_pret], dim=-1))

        # d. 注入显式的突变信息
        b = self.mut_bias(batch['mut_flag'].long())
        x = x + b

        # e. 编码配对特征并通过可训练的图注意力网络
        A = self.pair_atoms
        z = self.pair_encoder(
            aa = batch['aa'],
            res_nb = batch['res_nb'], chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'][:, :, :A], mask_atoms = batch['mask_heavyatom'][:, :, :A],
        )
        x = self.attn_encoder(
            pos_atoms = batch['pos_heavyatom'], 
            res_feat = x, pair_feat = z, 
            mask = mask_residue
        )
        return x

    def forward(self, batch):
        """ [移植] 仿照 rde_ddg.py 的对称化前向传播与损失计算 """
        # 若提供了预计算骨干上下文，则分别喂给 WT / MT 分支（见 encode 中的说明）
        aa_wt = batch['aa_wt'] if 'aa_wt' in batch else batch['aa']
        batch_wt = {**batch, 'aa': aa_wt, 'x_pret': batch.get('x_pret_wt', None)}
        batch_mt = {**batch, 'aa': batch['aa_mut'], 'x_pret': batch.get('x_pret_mt', None)}

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)

        if self.rich_head:
            # 多尺度池化：global / local(突变) / env(突变邻域)。WT、MT 共用同一组掩码。
            masks = self._scale_masks(batch)
            F_mt, F_wt = self._pool_feats(h_mt, masks), self._pool_feats(h_wt, masks)
            # 读出输入 = [A三类, B三类, (A-B)三类]；交换 A/B 即可得到反向分支。
            fwd = self.ddg_readout(torch.cat([F_mt, F_wt, F_mt - F_wt], dim=-1)).squeeze(-1)
            rev = self.ddg_readout(torch.cat([F_wt, F_mt, F_wt - F_mt], dim=-1)).squeeze(-1)
        else:
            if self.pool_mode == 'global_max':
                residue_mask = (
                    batch['mask'] if 'mask' in batch else (aa_wt != 0)
                ).bool()
                H_mt = self._masked_max(h_mt, residue_mask)
                H_wt = self._masked_max(h_wt, residue_mask)
            elif self.pool_mode == 'mutation_mean':
                mutation_mask = batch['mut_flag'].bool()
                H_mt = self._masked_mean(h_mt, mutation_mask)
                H_wt = self._masked_mean(h_wt, mutation_mask)
            else:
                patch = self._patch_mask(batch)
                H_mt = self._masked_mean(h_mt, patch)
                H_wt = self._masked_mean(h_wt, patch)
            fwd = self.ddg_readout(H_mt - H_wt).squeeze(-1)
            rev = self.ddg_readout(H_wt - H_mt).squeeze(-1)

        y = batch['ddG'].view_as(fwd)
        loss = self.loss_from_branches(fwd, rev, y)

        # 输出严格反对称：交换 wt/mt 会交换 fwd 与 rev，使输出精确取负。
        ddg_out = 0.5 * (fwd - rev)

        loss_dict = {'regression': loss, 'total_loss': loss}
        out_dict = {'ddg': ddg_out}
        if self.aggregate_loss_across_accumulation and self.training:
            out_dict.update({
                '_loss_fwd': fwd,
                '_loss_rev': rev,
                '_loss_target': y,
            })
        return loss_dict, out_dict

def model_factory(cfg) -> RDEDeltaDeltaG:
    logger.info("Initializing the FULLY-UPGRADED RDEDeltaDeltaG model...")
    return RDEDeltaDeltaG(cfg)
