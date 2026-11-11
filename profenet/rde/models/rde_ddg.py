# rde/models/rde_ddg.py
# --- 完全重构版 ---
# 1. 采用高效的“向量化”特征聚合，移除了原有的for循环。
# 2. 实施“部分微调”策略，解冻骨干网络的顶层以适应下游任务。
# 3. 简化预测头，降低过拟合风险。
# 4. 保留了优秀的对称混合损失设计。

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# 假设 EnhancedRDE 模型位于指定路径
from rde.models.enhanced_rde import EnhancedRDE

logger = logging.getLogger(__name__)


# +++ 损失函数 (保留原设计) +++
class PearsonCorrelationLoss(nn.Module):
    """
    损失函数 L = 1 - Pearson's Correlation Coefficient.
    目标是最大化相关性，等价于最小化此损失。
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)

        vx = y_pred_flat - torch.mean(y_pred_flat)
        vy = y_true_flat - torch.mean(y_true_flat)

        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + self.eps) * torch.sqrt(torch.sum(vy ** 2) + self.eps))
        
        return 1 - corr

# +++ 工具函数 (保留原设计) +++
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    在指定维度上进行安全的 Masked Mean Pooling。
    Args:
        x (torch.Tensor): 输入张量，形状如 (B, N, D)。
        mask (torch.Tensor): 掩码张量，形状如 (B, N)。
    Returns:
        torch.Tensor: 池化后的张量，形状如 (B, D)。
    """
    mask = mask.bool().unsqueeze(-1)
    x_masked = x.masked_fill(~mask, 0.0)
    sum_val = x_masked.sum(dim=1)
    count_val = mask.sum(dim=1).clamp(min=1e-9)
    return sum_val / count_val


# --- 核心模型：DDG_RDE_Network ---
class DDG_RDE_Network(nn.Module):
    """
    使用预训练的 EnhancedRDE 骨干网络进行ΔΔG预测的微调模型。
    
    核心修改:
    1. 特征提取后，使用全局差异池化，而非复杂的循环。
    2. 骨干网络采用部分解冻的微调策略，以更好地适应任务。
    3. 预测头被简化为一个标准的MLP，接收WT、MT及其差异的全局特征。
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.rde_features: Optional[torch.Tensor] = None
        
        # --- 1. 加载并配置预训练的骨干网络 ---
        logger.info("正在加载预训练的 EnhancedRDE 骨干网络...")
        ckpt_path = cfg.model.rde_checkpoint
        state = torch.load(ckpt_path, map_location="cpu")
        pretrain_config = state.get("config", {})
        
        self.rde = EnhancedRDE(pretrain_config)
        self.rde.load_state_dict(state.get("model_state_dict", state), strict=False)
        
        # --- 2. 【修改】实施部分微调策略 (Partial Fine-tuning) ---
        # 默认冻结所有参数
        for param in self.rde.parameters():
            param.requires_grad_(False)
        
        # 解冻顶部的层，使其能够适应新的ΔΔG任务
        # 这是一个超参数，可以根据实验调整
        num_fusion_layers_to_unfreeze = 2 
        
        logger.info(f"解冻骨干网络的最后 {num_fusion_layers_to_unfreeze} 个 FusionLayer 和整个 StructureModule 进行微调。")
        for layer in self.rde.fusion_stack[-num_fusion_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad_(True)

        for param in self.rde.structure_module.parameters():
            param.requires_grad_(True)
        
        # 注册钩子以提取中间层特征
        self._seq_hook_handle = self.rde.output_norm.register_forward_hook(self._seq_hook_fn())
        
        # --- 3. 【修改】构建简化的ΔΔG预测头 ---
        self.seq_dim = self.rde.hidden_dim
        dropout = float(getattr(cfg.model, "dropout", 0.1))
        readout_dim = int(getattr(cfg.model, "readout_dim", 512))

        # 预测头接收3部分拼接的特征：[MT全局特征, WT全局特征, 差异全局特征]
        head_input_dim = self.seq_dim * 3
        
        logger.info(f"构建简化的MLP预测头，输入维度: {head_input_dim}")
        self.ddg_predictor_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, readout_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, 1),
        )
        
        # --- 4. 配置损失函数 (保留原设计) ---
        self.regression_criterion = nn.MSELoss()
        self.correlation_criterion = PearsonCorrelationLoss()
        self.loss_alpha = float(getattr(cfg.model, "loss_alpha", 0.2)) # MSE损失的权重
        logger.info(f"使用混合损失函数 (MSE + Pearson)，MSE权重 alpha = {self.loss_alpha}")

    def _seq_hook_fn(self):
        """钩子函数，用于捕获RDE骨干网络输出的最终序列特征。"""
        def hook(_module, _inp, out):
            self.rde_features = out
        return hook

    def _encode_once(self, batch_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """单次调用RDE骨干网络进行编码。"""
        # 确保解冻的层处于训练模式（如果模型在.train()状态下）
        self.rde.train(self.training)
        
        self.rde_features = None
        # RDE模型返回(pred_chi_sincos, pred_coord_update), aux_preds，我们这里只用它来触发钩子
        with torch.set_grad_enabled(self.training): # 在训练时开启梯度，以便微调
            _ = self.rde(batch_dict)
        
        if self.rde_features is None:
            raise RuntimeError("使用钩子捕获RDE特征失败。")
        
        return self.rde_features, batch_dict["mask"]

    def forward(self, batch: Dict):
        # 准备WT和MT的输入字典
        wt_sub = {"aa": batch["aa_wt"], "mask": batch["mask"], "pos_heavyatom": batch["pos_heavyatom"]}
        mt_sub = {"aa": batch["aa_mut"], "mask": batch["mask"], "pos_heavyatom": batch["pos_heavyatom"]}

        # --- 【修改】高效的向量化特征提取 ---
        # 1. 分别编码野生型(WT)和突变型(MT)
        wt_tok, wt_mask = self._encode_once(wt_sub)
        mt_tok, mt_mask = self._encode_once(mt_sub)

        # 2. 计算全局特征表示
        global_wt_feat = masked_mean(wt_tok, wt_mask)
        global_mt_feat = masked_mean(mt_tok, mt_mask)
        
        # 3. 计算核心的“差异”特征
        seq_feat_diff = mt_tok - wt_tok
        global_diff_feat = masked_mean(seq_feat_diff, wt_mask)

        # 4. 拼接成最终的特征向量
        final_vec_fwd = torch.cat([global_mt_feat, global_wt_feat, global_diff_feat], dim=-1)
        final_vec_rev = torch.cat([global_wt_feat, global_mt_feat, -global_diff_feat], dim=-1)

        # --- 5. 预测ΔΔG (保留对称预测) ---
        pred_forward = self.ddg_predictor_head(final_vec_fwd).squeeze(-1) # MT - WT, 趋向于 ddG
        pred_reverse = self.ddg_predictor_head(final_vec_rev).squeeze(-1) # WT - MT, 趋向于 -ddG
        
        # 使用反对称平均，使预测更稳定、更符合物理直觉
        ddg_pred = 0.5 * (pred_forward - pred_reverse)

        # --- 6. 计算损失 (保留对称混合损失) ---
        loss_dict = {}
        if "ddG" in batch and batch["ddG"] is not None:
            y_true = batch["ddG"].view_as(ddg_pred)
            
            # 正向损失 (MT-WT -> ddG)
            loss_mse_fwd = self.regression_criterion(pred_forward, y_true)
            loss_corr_fwd = self.correlation_criterion(pred_forward, y_true)
            loss_fwd = self.loss_alpha * loss_mse_fwd + (1 - self.loss_alpha) * loss_corr_fwd

            # 反向损失 (WT-MT -> -ddG)
            loss_mse_rev = self.regression_criterion(pred_reverse, -y_true)
            loss_corr_rev = self.correlation_criterion(pred_reverse, -y_true)
            loss_rev = self.loss_alpha * loss_mse_rev + (1 - self.loss_alpha) * loss_corr_rev
            
            # 对称平均总损失
            loss = (loss_fwd + loss_rev) / 2.0
            
            loss_dict["total_loss"] = loss
            loss_dict["mse_loss"] = (loss_mse_fwd.detach() + loss_mse_rev.detach()) / 2.0
            loss_dict["corr_loss"] = (loss_corr_fwd.detach() + loss_corr_rev.detach()) / 2.0

        out = {"ddG_pred": ddg_pred, "ddG_true": batch.get("ddG", None)}
        return loss_dict, out
    
    def __del__(self):
        # 清理钩子，防止内存泄漏
        if hasattr(self, '_seq_hook_handle') and self._seq_hook_handle:
            self._seq_hook_handle.remove()


def model_factory(cfg, **kwargs):
    return DDG_RDE_Network(cfg)
