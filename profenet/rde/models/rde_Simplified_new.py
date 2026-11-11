# rde_simplified.py
# (增加了 sample 和 pack 方法以支持评估)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, List
import numpy as np # 需要 numpy

# ... [之前的所有辅助模块和类保持不变] ...
# ... [All previous helper modules and classes remain unchanged] ...
# GVP, RBFExpansion, GVPMessageLayer, GVPGeometryEncoder, OuterProductMean, 
# TriangleAttention, AxialAttention, HybridEncoderBlock...

# 假设这些流模型模块在您的项目路径中可用
from rde.modules.flows.spline import ContextualCircularSplineFlow
from rde.modules.flows.coupling import ContextualCircularSplineCouplingLayer
from rde.modules.flows.container import ContextualSequentialFlow

class BBHeavyAtom:
    N, CA, C, O, CB = 0, 1, 2, 3, 4
num_aa_types = 21

class GVP(nn.Module):
    def __init__(self, s_in, v_in, s_out, v_out, drop=0.1):
        super().__init__()
        self.Ws = nn.Linear(s_in + v_in, s_out)
        self.Wv = nn.Parameter(torch.randn(v_out, v_in) * (1 / math.sqrt(max(1, v_in))))
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)
        self.gate_s = nn.Linear(s_out, s_out)
        self.gate_v = nn.Linear(s_out, v_out)

    @staticmethod
    def _project_vec(v: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...ic,oi->...oc", v, W)

    def forward(self, s: torch.Tensor, v: torch.Tensor):
        v_norm = torch.linalg.vector_norm(v, dim=-1)
        s_cat = torch.cat([s, v_norm], dim=-1)
        s_out = self.act(self.Ws(s_cat))
        v_lin = self._project_vec(v, self.Wv)
        gs = torch.sigmoid(self.gate_s(s_out))
        gv = torch.sigmoid(self.gate_v(s_out)).unsqueeze(-1)
        s_out = gs * s_out
        v_out = gv * v_lin
        return self.drop(s_out), self.drop(v_out)

class RBFExpansion(nn.Module):
    def __init__(self, num_rbfs=16, cutoff=20.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, steps=num_rbfs)
        widths = torch.full_like(centers, (cutoff / num_rbfs))
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, d):
        d = d.unsqueeze(-1)
        return torch.exp(-((d - self.centers) ** 2) / (2 * (self.widths ** 2 + 1e-8)))

class GVPMessageLayer(nn.Module):
    def __init__(self, s_dim, v_dim, s_edge_dim, v_hidden_dim=8, drop=0.1):
        super().__init__()
        self.v_hidden_dim = v_hidden_dim
        self.edge_mlp = nn.Sequential(nn.Linear(s_edge_dim, s_dim), nn.SiLU(), nn.Linear(s_dim, s_dim))
        self.dir_mlp = nn.Sequential(nn.Linear(3, s_dim), nn.SiLU(), nn.Linear(s_dim, s_dim), nn.SiLU())
        self.msg_gvp = GVP(s_in=s_dim, v_in=v_dim, s_out=s_dim, v_out=v_hidden_dim, drop=drop)
        self.update_gvp = GVP(s_in=s_dim * 2, v_in=v_hidden_dim, s_out=s_dim, v_out=v_dim, drop=drop)
        self.norm_s = nn.LayerNorm(s_dim)
        self.drop_path = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, s, v, x, edge_index, rbf_layer: RBFExpansion):
        src, dst = edge_index
        
        s_agg = torch.zeros_like(s)
        v_agg = torch.zeros(s.shape[0], self.v_hidden_dim, 3, device=v.device, dtype=v.dtype)

        if edge_index.numel() > 0:
            disp = x[dst] - x[src]
            dist = torch.norm(disp, dim=-1)
            rbf_feat = rbf_layer(dist)
            dir_feat = self.dir_mlp(F.normalize(disp, dim=-1))
            edge_feat = self.edge_mlp(rbf_feat) + dir_feat
            
            s_msg, v_msg = self.msg_gvp(s[src], v[src])
            s_msg = s_msg + edge_feat
            
            s_agg.index_add_(0, dst, s_msg)
            v_agg.index_add_(0, dst, v_msg)

        s_out, v_out = self.update_gvp(torch.cat([s, s_agg], dim=-1), v_agg)
        s_out = self.norm_s(s + self.drop_path(s_out))
        return s_out, v_out

class GVPGeometryEncoder(nn.Module):
    def __init__(self, s_in_dim, s_hidden_dim, v_dim=16, layers=3, drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.rbf = RBFExpansion(num_rbfs=16, cutoff=20.0)
        self.input_proj = nn.Linear(s_in_dim, s_hidden_dim)
        for _ in range(layers):
            self.layers.append(GVPMessageLayer(s_hidden_dim, v_dim, s_edge_dim=16, drop=drop))
        self.out_norm = nn.LayerNorm(s_hidden_dim)

    def forward(self, h, x, edge_index, v_init):
        s = self.input_proj(h)
        v = v_init
        for layer in self.layers:
            s, v = layer(s, v, x, edge_index, self.rbf)
        return self.out_norm(s)

class OuterProductMean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.proj_a = nn.Linear(in_dim, 32)
        self.proj_b = nn.Linear(in_dim, 32)
        self.out_proj = nn.Linear(32 * 32, out_dim)

    def forward(self, x, mask):
        x = self.ln(x)
        a = self.proj_a(x)
        b = self.proj_b(x)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            a, b = a * mask, b * mask
        outer = torch.einsum("b i c, b j d -> b i j c d", a, b)
        outer = rearrange(outer, "b i j c d -> b i j (c d)")
        return self.out_proj(outer)

class TriangleAttention(nn.Module):
    def __init__(self, pair_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, pair_dim // n_heads
        self.norm = nn.LayerNorm(pair_dim)
        self.to_qkv = nn.Linear(pair_dim, self.head_dim * n_heads * 3, bias=False)
        self.output_proj = nn.Linear(self.head_dim * n_heads, pair_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, mask):
        b, n, _, d = z.shape
        z_norm = self.norm(z)
        attn_mask_2d = mask[:, :, None] * mask[:, None, :]
        q, k, v = self.to_qkv(z_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b i j (h d) -> b h i j d", h=self.n_heads), (q, k, v))
        q, k = q * (self.head_dim ** -0.25), k * (self.head_dim ** -0.25)
        attn_logits = torch.einsum("b h i k d, b h j k d -> b h i j", q, k)
        attn_mask_4d = rearrange(attn_mask_2d, "b i j -> b 1 i j")
        attn_logits.masked_fill_(~attn_mask_4d, -torch.finfo(attn_logits.dtype).max)
        attn = F.softmax(attn_logits, dim=-1)
        updates = torch.einsum("b h i j, b h i k d -> b h j k d", attn, v)
        updates = rearrange(updates, "b h i j d -> b i j (h d)")
        return self.dropout(self.output_proj(updates))

class AxialAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super().__init__()
        self.heads, self.dim_head = heads, in_dim // heads
        self.scale = self.dim_head ** -0.5
        self.ln = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.to_out = nn.Linear(in_dim, out_dim)

    def forward(self, x, mask=None):
        x = self.ln(x)
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        attn_scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            mask_2d = mask[:, None, None, :]
            attn_scores.masked_fill_(~mask_2d, -torch.finfo(attn_scores.dtype).max)
        attn = F.softmax(attn_scores, dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class HybridEncoderBlock(nn.Module):
    def __init__(self, s_dim, z_dim, n_heads, dropout):
        super().__init__()
        self.opm = OuterProductMean(s_dim, z_dim)
        self.tri_attn_row = TriangleAttention(z_dim, n_heads, dropout)
        self.tri_attn_col = TriangleAttention(z_dim, n_heads, dropout)
        self.ffn_pair = nn.Sequential(nn.LayerNorm(z_dim), nn.Linear(z_dim, z_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(z_dim * 4, z_dim))
        self.axial_attn = AxialAttention(in_dim=z_dim, out_dim=s_dim, heads=n_heads)
        self.ffn_seq = nn.Sequential(nn.LayerNorm(s_dim), nn.Linear(s_dim, s_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(s_dim * 4, s_dim))

    def forward(self, s, z, mask):
        s_res, z_res = s, z
        z = z + self.opm(s, mask)
        z = z + self.tri_attn_row(z, mask)
        z = rearrange(z, 'b i j d -> b j i d')
        z = z + self.tri_attn_col(z, mask)
        z = rearrange(z, 'b j i d -> b i j d')
        z = z_res + self.ffn_pair(z)
        s_update_from_z = self.axial_attn(z.mean(dim=2), mask)
        s = s + s_update_from_z
        s = s_res + self.ffn_seq(s)
        return s, z

def _build_flow(num_chis, n_context_dims, cfg_flow):
    blocks = []
    spline_params = cfg_flow['spline_params']
    if num_chis == 1:
        for _ in range(cfg_flow['num_blocks']):
            blocks.append(ContextualCircularSplineFlow(n_context_dims, **spline_params))
    else:
        for i in range(cfg_flow['num_blocks']):
            blocks.append(ContextualCircularSplineCouplingLayer(
                n_dims=num_chis, 
                mapping_dims=[i % num_chis], 
                n_context_dims=n_context_dims, 
                **spline_params
            ))
    return ContextualSequentialFlow(blocks)

def _latent_log_prob(z, num_chis):
    volume = (2 * math.pi) ** num_chis
    logp = math.log(1 / volume)
    return torch.full(z.shape[:-1], logp, device=z.device, dtype=torch.float)

# --- 新增辅助函数 (适配自 rde.py) ---
def sample_latent(shape, num_chis, device):
    """从均匀分布中采样潜在变量 z"""
    shape = [*shape, num_chis]
    # 采样范围为 [-pi, pi]
    z = torch.rand(shape, device=device) * (2 * np.pi) - np.pi
    return z


class RDE_Simplified(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m_cfg = cfg['model']
        self.aa_embed = nn.Embedding(num_aa_types, m_cfg['s_dim'])
        self.gvp_v_dim = m_cfg['gvp']['v_dim']
        self.geometry_encoder = GVPGeometryEncoder(
            m_cfg['s_dim'], m_cfg['s_dim'], self.gvp_v_dim, m_cfg['gvp']['layers'], m_cfg['dropout']
        )
        self.pair_dist_embed = nn.Sequential(RBFExpansion(16, 20.0), nn.Linear(16, m_cfg['z_dim']))
        self.encoder_blocks = nn.ModuleList([
            HybridEncoderBlock(m_cfg['s_dim'], m_cfg['z_dim'], m_cfg['n_heads'], m_cfg['dropout'])
            for _ in range(m_cfg['num_blocks'])
        ])
        self.flows = nn.ModuleList([_build_flow(i, m_cfg['s_dim'], m_cfg['flow']) for i in range(1, 5)])
        self.register_buffer('num_chis_of_aa', self._get_num_chis_of_aa())

    def _get_num_chis_of_aa(self):
        data = [0, 1, 2, 2, 2, 0, 2, 2, 4, 2, 3, 2, 0, 3, 4, 1, 1, 1, 2, 2, 0] 
        return torch.tensor(data, dtype=torch.long)

    @staticmethod
    def _build_graph(X: torch.Tensor, mask: torch.Tensor, k: int = 16):
        try: from torch_geometric.nn import knn_graph
        except ImportError: raise ImportError("请安装 torch_geometric 以构建 k-NN 图。")
        B, L, _ = X.shape
        edges = []
        for b in range(B):
            valid_indices = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if valid_indices.numel() < 2: continue
            pos = X[b, valid_indices]
            edge = knn_graph(pos, k=min(k, pos.size(0) - 1), loop=False)
            edge = valid_indices[edge]
            if b > 0: edge = edge + b * L
            edges.append(edge)
        return torch.cat(edges, dim=1) if edges else torch.empty((2, 0), dtype=torch.long, device=X.device)

    @staticmethod
    def _get_backbone_local_frames(pos_heavyatom: torch.Tensor) -> torch.Tensor:
        pos_n,pos_ca,pos_c = [pos_heavyatom[:,:,i,:] for i in [BBHeavyAtom.N,BBHeavyAtom.CA,BBHeavyAtom.C]]
        v_ca_c, v_ca_n = pos_c - pos_ca, pos_n - pos_ca
        e_x = F.normalize(v_ca_c, dim=-1)
        e_z = F.normalize(torch.cross(v_ca_n, v_ca_c, dim=-1), dim=-1)
        e_y = torch.cross(e_z, e_x, dim=-1)
        return torch.nan_to_num(torch.stack([e_x, e_y, e_z], dim=2))
        
    def _mle_loss(self, batch, logprobs_by_nchis: List[torch.Tensor]):
        n_chis_data = batch['chi_mask'].sum(-1)
        chi_complete = batch.get('chi_complete', batch['chi_mask'].all(dim=-1))
        total_loss, total_res = 0.0, 0.0
        for n_chis in range(1, 5):
            logprob = logprobs_by_nchis[n_chis-1]
            mask = chi_complete & (n_chis_data == n_chis)
            if mask.any():
                loss, count = -logprob[mask].sum(), mask.sum().float()
                total_loss, total_res = total_loss + loss, total_res + count
        return total_loss / (total_res + 1e-8)

    # --- 新增 encode 方法 ---
    # 用于提取上下文特征 c, 是 forward 的一部分，但在 sample 中被独立调用
    def encode(self, batch: Dict[str, torch.Tensor]):
        aa, mask, pos_heavyatom = batch['aa'], batch['mask'], batch['pos_atoms']
        pos_alpha = pos_heavyatom[:, :, BBHeavyAtom.CA, :]
        b, n = aa.shape

        s_initial = self.aa_embed(aa)
        local_frames = self._get_backbone_local_frames(pos_heavyatom)
        v_init = torch.zeros(b, n, self.gvp_v_dim, 3, device=aa.device, dtype=pos_alpha.dtype)
        v_init[:, :, :min(3, self.gvp_v_dim), :] = local_frames[:, :, :min(3, self.gvp_v_dim), :]
        
        edge_index = self._build_graph(pos_alpha, mask)
        s = self.geometry_encoder(
            s_initial.reshape(b*n, -1), 
            pos_alpha.reshape(b*n, 3), 
            edge_index, 
            v_init.reshape(b*n, self.gvp_v_dim, 3)
        ).reshape(b, n, -1)
        
        pair_dist = torch.cdist(pos_alpha, pos_alpha)
        z = self.pair_dist_embed(pair_dist)
        
        for block in self.encoder_blocks:
            s, z = block(s, z, mask)
        return s # 返回最终的上下文特征

    def forward(self, batch: Dict[str, torch.Tensor]):
        s_context = self.encode(batch)
        loss_dict, pred_dict = {}, {}

        logprobs_by_nchis = []
        for i in range(1, 5):
            chi_native = batch['chi_native'][:, :, :i]
            z_flow, logabsdet = self.flows[i-1](chi_native, s_context, inverse=False)
            logp = _latent_log_prob(z_flow, i) + logabsdet
            logprobs_by_nchis.append(logp)
        
        L_density = self._mle_loss(batch, logprobs_by_nchis)
        loss_dict['density'] = L_density
        loss_dict['total'] = L_density
        pred_dict['log_probs'] = logprobs_by_nchis
        
        return loss_dict, pred_dict

    # --- 新增 sample 方法 (适配自 rde.py) ---
    def sample(self, batch, n_samples=1):
        c = self.encode(batch) # 上下文特征

        # 如果需要多个样本，扩展上下文张量的维度
        if n_samples > 1:
            c_shape = [n_samples] + list(c.shape)
            c = c.unsqueeze(0).expand(c_shape)
        
        # 1. 从潜在空间采样 z
        zs = [
            sample_latent(c.shape[:-1], n_chis, device=c.device)
            for n_chis in range(1, 4+1)
        ]
        logpzs = [
            _latent_log_prob(zs[n_chis-1], n_chis)
            for n_chis in range(1, 4+1)
        ]

        # 2. 使用流模型的逆变换，将 z 映射回角度空间 x
        # --- 核心修复点在这里：确保列表被初始化 ---
        xs, logprobs = [], []
        # --- 修复结束 ---
        for n_chis in range(1, 4+1):
            x, logabsdet = self.flows[n_chis-1](zs[n_chis-1], c, inverse=True)
            x = F.pad(x, pad=(0, 4-n_chis), value=0) # 填充到4个维度
            logpx = logpzs[n_chis-1] - logabsdet
            xs.append(x)
            logprobs.append(logpx)
        
        # 将不同 chi 数量的结果堆叠起来
        xs = torch.stack(xs, dim=-1)
        logprobs = torch.stack(logprobs, dim=-1)

        # 根据每个氨基酸的实际 chi 数量，选择对应的预测结果
        n_chis_all = self.num_chis_of_aa[batch['aa']] # [B, L]
        
        # (整合了之前的修复) 将 chi 数量 (1-4) 映射到索引 (0-3)
        idx = (n_chis_all - 1).clamp(min=0).long()
        
        if n_samples > 1:
            idx = idx.unsqueeze(0).expand(n_samples, -1, -1)
        
        # xs_final: [S, B, L, 4]
        xs_final = torch.gather(
            xs, dim=-1, 
            index=idx[..., None, None].expand(-1, -1, -1, 4, 1)
        ).squeeze(-1)
        # logprobs_final: [S, B, L]
        logprobs_final = torch.gather(
            logprobs, dim=-1,
            index=idx[..., None].expand(-1, -1, -1, 1)
        ).squeeze(-1)

        # 对于没有chi角的残基，其预测值和logprob应该为0
        mask_no_chi = (n_chis_all == 0) # 形状: [B, L]
        if n_samples > 1:
            # (整合了之前的修复) 将掩码扩展以匹配样本维度 S
            mask_no_chi = mask_no_chi.unsqueeze(0).expand(n_samples, -1, -1) # 得到形状 [S, B, L]

        xs_final[mask_no_chi] = 0.
        logprobs_final[mask_no_chi] = 0.

        return xs_final, logprobs_final

    # --- 新增 pack 方法 (适配自 rde.py) ---
    def pack(self, batch, n_samples=50):
        # 采样 n_samples 个候选者
        xs, logprobs = self.sample(batch, n_samples=n_samples)
        
        # 选取每个位置上 log-probability 最高的样本
        logprobs_max, smp_idx = logprobs.max(dim=0) # [B, L]
        
        # 整理索引以便 gather
        # smp_idx [B, L] -> [1, B, L, 1] -> [1, B, L, 4]
        smp_idx = smp_idx.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 4)
        
        # 提取最优样本
        xs_packed = torch.gather(xs, dim=0, index=smp_idx).squeeze(0)
        
        return xs_packed, logprobs_max