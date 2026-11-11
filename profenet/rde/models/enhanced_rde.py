# rde/models/enhanced_rde.py — Integrated Multi-Atom IPA Version
# - Multi-atom IPA returns [B, N, M, 3] (N/CA/C/O/CB...)
# - Config switch: model.use_chi_hint (default True); set False at eval to disable chi hints
# - Drop-in compatible with existing train/eval scripts that expect (pred_chi_sincos, pred_coord_update)

import math
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch_geometric.nn import radius_graph, knn_graph

# Assumes BBHeavyAtom has N=0, CA=1, C=2, O=3, CB=4 indices.
from rde.utils.chi_context import encode_chi_hint
from rde.utils.protein.constants import BBHeavyAtom, AA

logger = logging.getLogger(__name__)


# ----------------------------
# Utilities
# ----------------------------
def normalize_angle(theta: torch.Tensor) -> torch.Tensor:
    return (theta + math.pi) % (2 * math.pi) - math.pi


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, freqs):
    return (x * freqs.cos()) + (rotate_half(x) * freqs.sin())


class RBFExpansion(nn.Module):
    def __init__(self, num_rbfs=16, cutoff=20.0):
        super().__init__()
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, steps=num_rbfs)
        widths = torch.full_like(centers, (cutoff / num_rbfs))
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, d):
        d = d.unsqueeze(-1)
        return torch.exp(-((d - self.centers) ** 2) / (2 * (self.widths ** 2 + 1e-8)))


# ----------------------------
# GVP primitives
# ----------------------------
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


class GVPMessageLayer(nn.Module):
    def __init__(self, s_dim, v_dim, s_edge_dim, v_hidden_dim=8, drop=0.1):
        super().__init__()
        self.v_hidden_dim = int(v_hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(s_edge_dim, s_dim), nn.SiLU(), nn.Linear(s_dim, s_dim)
        )
        self.dir_mlp = nn.Sequential(
            nn.Linear(3, s_dim), nn.SiLU(), nn.Linear(s_dim, s_dim), nn.SiLU()
        )
        self.msg_gvp = GVP(s_in=s_dim, v_in=v_dim, s_out=s_dim, v_out=v_hidden_dim, drop=drop)
        self.update_gvp = GVP(s_in=s_dim * 2, v_in=v_hidden_dim, s_out=s_dim, v_out=v_dim, drop=drop)
        self.norm_s = nn.LayerNorm(s_dim)
        self.drop_path = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, s, v, x, edge_index, rbf_layer: RBFExpansion):
        src, dst = edge_index
        if edge_index.numel() > 0:
            disp = x[dst] - x[src]
            dist = torch.norm(disp, dim=-1)
            rbf_feat = rbf_layer(dist)
            dir_feat = self.dir_mlp(F.normalize(disp, dim=-1))
            edge_feat = self.edge_mlp(rbf_feat) + dir_feat
            s_msg, v_msg = self.msg_gvp(s[src], v[src])
            s_msg = s_msg + edge_feat
            s_agg = torch.zeros_like(s)
            v_agg = v.new_zeros((v.shape[0], self.v_hidden_dim, v.shape[-1]))
            s_agg.index_add_(0, dst, s_msg)
            v_agg.index_add_(0, dst, v_msg)
        else:
            s_agg = torch.zeros_like(s)
            v_agg = v.new_zeros((v.shape[0], self.v_hidden_dim, v.shape[-1]))
        s_out, v_out = self.update_gvp(torch.cat([s, s_agg], dim=-1), v_agg)
        s_out = self.norm_s(s + self.drop_path(s_out))
        return s_out, v_out


# ----------------------------
# GVP Geometry Encoder
# ----------------------------
class GVPGeometryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, v_dim=8, layers=3, drop=0.1):
        super().__init__()
        self.s_hidden = hidden_dim
        self.v_dim = v_dim
        self.layers = nn.ModuleList()
        self.rbf = RBFExpansion(num_rbfs=16, cutoff=20.0)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        for _ in range(layers):
            self.layers.append(GVPMessageLayer(hidden_dim, v_dim, s_edge_dim=16, drop=drop))
        self.out_norm = nn.LayerNorm(self.s_hidden)

    def forward(self, h, x, edge_index, v_init):
        s = self.input_proj(h)
        v = v_init  # initial local frames
        for layer in self.layers:
            s, v = layer(s, v, x, edge_index, self.rbf)
        return self.out_norm(s)


# ----------------------------
# Attention + FFN Blocks
# ----------------------------
def _masked_softmax(logits, mask, dim=-1):
    if mask is None:
        return F.softmax(logits, dim=dim)
    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
    probabilities = F.softmax(masked_logits, dim=dim)
    return torch.where(mask, probabilities, torch.zeros_like(probabilities))


class OptimizedAxialAttention(nn.Module):
    """
    AxialAttention 的优化版实现。
    使用 torch.nn.functional.scaled_dot_product_attention (SDPA) 来避免
    实例化巨大的 (B*L, H, L, L) 注意力矩阵，从而根本上解决O(L^3)内存问题。
    需要 PyTorch 2.0 或更高版本。
    """
    def __init__(self, in_dim, out_dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5 # SDPA 会自动处理 scale，但保留它以维持接口一致性
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(in_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, out_dim)

    def forward(self, x, mask=None):
        # x 的形状: (B*L, L, D_in)
        b, n, _ = x.shape

        # 1. 计算 Q, K, V 并重塑以适应多头注意力
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # 2. 准备 attention mask (如果提供)
        # SDPA 需要一个布尔掩码，其中 False 代表需要被 mask 的位置
        if mask is not None:
            # mask 形状: (B*L, L), 需变为 (B*L, 1, 1, L) 以便广播
            safe_mask = mask.bool()
            valid_rows = safe_mask.any(dim=-1)
            if (~valid_rows).any():
                safe_mask = safe_mask.clone()
                safe_mask[~valid_rows, 0] = True
            attn_mask = safe_mask.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None
            valid_rows = None

        # 3. 调用高度优化的 SDPA 函数
        # 这是一个融合操作，不会在内存中创建完整的 (N, N) 注意力矩阵
        # dropout 和 softmax 也在此函数内部完成
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            # dropout_p=self.dropout_p if self.training else 0.0 # 如果需要dropout
        )

        # 4. 重塑输出并应用最终的线性层
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if valid_rows is not None:
            out = torch.where(valid_rows[:, None, None], out, torch.zeros_like(out))
        return out


class AxialAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(in_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, out_dim)

    def forward(self, x, mask=None):
        b, j, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b j (h d) -> b h j d", h=self.heads), qkv)
        attn_scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            valid_rows = mask.bool().any(dim=-1)
            key_mask = rearrange(mask.bool(), "b j -> b 1 1 j")
        else:
            valid_rows = None
            key_mask = None
        attn = _masked_softmax(attn_scores, key_mask, dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h j d -> b j (h d)")
        out = self.to_out(out)
        if valid_rows is not None:
            out = torch.where(valid_rows[:, None, None], out, torch.zeros_like(out))
        return out


class OuterProductMean(nn.Module):
    def __init__(self, in_dim, out_dim, bottleneck):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, bottleneck), nn.GELU(), nn.Linear(bottleneck, in_dim)
        )
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x, mask=None):
        x = self.bottleneck(x)
        outer = torch.einsum("b i d, b j d -> b i j d", x, x)
        return self.proj(outer)


class TriangleAttention(nn.Module):
    def __init__(self, pair_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, pair_dim // n_heads
        self.norm = nn.LayerNorm(pair_dim)
        self.to_qkv = nn.Linear(pair_dim, self.head_dim * n_heads * 3, bias=False)
        self.output_proj = nn.Linear(self.head_dim * n_heads, pair_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, z, mask):
        b, n, _, d = z.shape
        z_norm = self.norm(z)
        attn_mask_2d = mask[:, :, None] * mask[:, None, :]
        q, k, v = self.to_qkv(z_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b i j (h d) -> b h i j d", h=self.n_heads), (q, k, v))
        i_pos = self.rope(n, z.device, z.dtype)
        j_pos = self.rope(n, z.device, z.dtype)
        q = apply_rotary_pos_emb(q, i_pos)
        k = apply_rotary_pos_emb(k, j_pos)
        k_mask = mask[:, None, None, :, None]
        q = q * k_mask
        k = k * k_mask
        q = q * (self.head_dim ** -0.25)
        k = k * (self.head_dim ** -0.25)
        attn_logits = torch.einsum("b h i k d, b h j k d -> b h i j", q, k)
        attn_mask_4d = rearrange(attn_mask_2d, "b i j -> b 1 i j")
        attn = _masked_softmax(attn_logits, attn_mask_4d, dim=-1)
        attn = self.dropout(attn)
        updates = torch.einsum("b h i j, b h i j d -> b h i d", attn, v)
        updates = rearrange(updates, "b h i d -> b i (h d)")
        updates = self.output_proj(updates)
        updates = updates[:, :, None, :] + updates[:, None, :, :]
        return z + self.dropout(updates)


class TriangularMultiplicativeUpdate(nn.Module):
    def __init__(self, pair_dim: int, hidden: int, outgoing: bool = True, dropout: float = 0.1):
        super().__init__()
        self.outgoing = outgoing
        self.left = nn.Linear(pair_dim, hidden, bias=False)
        self.right = nn.Linear(pair_dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, pair_dim, bias=False)
        self.gate = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, pair_dim), nn.Sigmoid())
        self.ln = nn.LayerNorm(pair_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        Pl = self.left(P)
        Pr = self.right(P)
        if self.outgoing:
            upd = torch.einsum("b i k h, b k j h -> b i j h", Pl, Pr)
        else:
            Pt = P.transpose(1, 2)
            Pl_t = self.left(Pt)
            Pr_t = self.right(Pt)
            upd = torch.einsum("b i k h, b k j h -> b i j h", Pl_t, Pr_t)
        upd = self.proj(self.drop(upd))
        gate = self.gate(P)
        return self.ln(P + gate * upd)


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, hidden_dim // n_heads
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, query, context, attn_bias=None, mask=None):
        q = self.to_q(query)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.n_heads), (q, k, v))
        attn_logits = torch.einsum("b h i d, b h j d -> b h i j", q, k) * (self.head_dim ** -0.5)
        if attn_bias is not None:
            attn_logits = attn_logits + attn_bias
        mask_2d = rearrange(mask.bool(), "b j -> b 1 1 j") if mask is not None else None
        attn = _masked_softmax(attn_logits, mask_2d, dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        attn_output = self.output_proj(out)
        query = self.layer_norm1(query + attn_output)
        ffn_output = self.ffn(query)
        query = self.layer_norm2(query + ffn_output)
        if mask is not None:
            query = torch.where(mask[..., None].bool(), query, torch.zeros_like(query))
        return query


class FusionLayer(nn.Module):
    def __init__(self, hidden_dim, pair_dim, n_heads, dropout):
        super().__init__()
        dim_head = hidden_dim // n_heads
        self.axial_attn = OptimizedAxialAttention(in_dim=pair_dim, out_dim=hidden_dim, heads=n_heads, dim_head=dim_head)
        self.opm = OuterProductMean(hidden_dim, pair_dim, 32)
        self.tri_attn_row = TriangleAttention(pair_dim, n_heads, dropout)
        self.tri_attn_col = TriangleAttention(pair_dim, n_heads, dropout)
        self.tri_mul_out = TriangularMultiplicativeUpdate(pair_dim, pair_dim // 2, outgoing=True, dropout=dropout)
        self.tri_mul_in = TriangularMultiplicativeUpdate(pair_dim, pair_dim // 2, outgoing=False, dropout=dropout)
        self.pair_to_bias = nn.Linear(pair_dim, n_heads, bias=False)
        self.geo_fusion = CrossAttentionFusion(hidden_dim, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, seq_context, geo_context, pair_rep, mask):
        b, i, j, d = pair_rep.shape
        residue_mask = mask.bool()
        pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]

        def mask_pairs(value):
            return torch.where(pair_mask[..., None], value, torch.zeros_like(value))

        seq_context = torch.where(
            residue_mask[..., None], seq_context, torch.zeros_like(seq_context)
        )
        pair_rep = mask_pairs(pair_rep)
        pair_rep = mask_pairs(pair_rep + self.opm(seq_context, residue_mask))
        pair_rep = mask_pairs(self.tri_mul_out(pair_rep))
        pair_rep = mask_pairs(pair_rep + self.tri_attn_row(pair_rep, residue_mask))
        pair_rep = mask_pairs(self.tri_mul_in(pair_rep))
        pair_rep_T = rearrange(pair_rep, "b i j d -> b j i d")
        pair_rep_T = pair_rep_T + self.tri_attn_col(pair_rep_T, residue_mask)
        pair_rep = rearrange(pair_rep_T, "b j i d -> b i j d")
        pair_rep = mask_pairs(pair_rep)
        row_attn_input = rearrange(pair_rep, "b i j d -> (b i) j d")
        mask_for_axial = repeat(residue_mask, "b j -> (b i) j", i=i)
        row_attn = self.axial_attn(row_attn_input, mask_for_axial)
        row_attn = rearrange(row_attn, "(b i) j d -> b i j d", b=seq_context.shape[0])
        row_attn = torch.where(
            pair_mask[..., None], row_attn, torch.zeros_like(row_attn)
        )
        valid_count = residue_mask.sum(dim=1).clamp_min(1).to(row_attn.dtype)
        row_summary = row_attn.sum(dim=2) / valid_count[:, None, None]
        seq_context = seq_context + row_summary
        attn_bias = rearrange(self.pair_to_bias(pair_rep), "b i j h -> b h i j")
        seq_context = self.geo_fusion(
            seq_context,
            geo_context,
            attn_bias=attn_bias,
            mask=residue_mask,
        )
        seq_context = seq_context + self.ffn(seq_context)
        seq_context = torch.where(
            residue_mask[..., None], seq_context, torch.zeros_like(seq_context)
        )
        return seq_context, pair_rep


class InvariantPointAttention(nn.Module):
    def __init__(self, dim, heads, pair_dim, scalar_kv_dims, point_kv_dims):
        super().__init__()
        self.heads = heads
        self.to_scalar_q = nn.Linear(dim, heads * scalar_kv_dims, bias=False)
        self.to_scalar_k = nn.Linear(dim, heads * scalar_kv_dims, bias=False)
        self.to_scalar_v = nn.Linear(dim, heads * scalar_kv_dims, bias=False)
        self.to_point_q = nn.Linear(dim, heads * point_kv_dims * 3, bias=False)
        self.to_point_k = nn.Linear(dim, heads * point_kv_dims * 3, bias=False)
        self.to_point_v = nn.Linear(dim, heads * point_kv_dims * 3, bias=False)
        self.to_pair_bias = nn.Linear(pair_dim, heads, bias=False)
        self.to_out = nn.Linear(heads * (scalar_kv_dims + pair_dim + point_kv_dims * 3), dim)
        self.point_weight = nn.Parameter(torch.log(torch.exp(torch.ones(heads)) - 1.0))

    def forward(self, single, pair, rotations, translations, mask):
        b, n, h = single.shape[0], single.shape[1], self.heads
        q_s, k_s, v_s = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
            (self.to_scalar_q(single), self.to_scalar_k(single), self.to_scalar_v(single)),
        )
        q_p, k_p, v_p = map(
            lambda t: rearrange(t, "b n (h d c) -> b h n d c", h=h, c=3),
            (self.to_point_q(single), self.to_point_k(single), self.to_point_v(single)),
        )
        rot_q_p = torch.einsum("b i r c, b h i d c -> b h i d r", rotations, q_p)
        rot_k_p = torch.einsum("b j r c, b h j d c -> b h j d r", rotations, k_p)
        rot_v_p = torch.einsum("b j r c, b h j d c -> b h j d r", rotations, v_p)
        pair_bias = rearrange(self.to_pair_bias(pair), "b i j h -> b h i j")
        scalar_scores = torch.einsum("b h i d, b h j d -> b h i j", q_s, k_s)
        point_scores = -self.point_weight.view(1, h, 1, 1) * torch.sum(
            (rot_q_p.unsqueeze(3) - rot_k_p.unsqueeze(2)) ** 2, dim=(-1, -2)
        )
        attn_logits = scalar_scores + point_scores + pair_bias
        attn_mask = (
            mask[:, None, :, None].bool() & mask[:, None, None, :].bool()
            if mask is not None
            else None
        )
        attn = _masked_softmax(attn_logits, attn_mask, dim=-1)
        agg_scalar = torch.einsum("b h i j, b h j d -> b h i d", attn, v_s)
        agg_pair = torch.einsum("b h i j, b i j d -> b h i d", attn, pair)
        agg_point = torch.einsum("b h i j, b h j d r -> b h i d r", attn, rot_v_p)
        out = torch.cat([agg_scalar, agg_pair, agg_point.reshape(b, h, n, -1)], dim=-1)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if mask is not None:
            out = torch.where(mask[..., None].bool(), out, torch.zeros_like(out))
        return out


# ----------------------------
# Multi-Atom IPA
# ----------------------------
class IPAModule(nn.Module):
    """
    Multi-atom IPA:
      - Iteratively update CA translations (stable backbone motion)
      - Atom head regresses per-atom local offsets in the residue frame (N–CA–C)
      - Returns pred_coord_update: [B, N, M, 3] (M = number of atoms provided)
    """
    def __init__(self, dim, pair_dim, depth=2, heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm_seq": nn.LayerNorm(dim),
                        "norm_pair": nn.LayerNorm(pair_dim),
                        "ipa": InvariantPointAttention(dim, heads, pair_dim, scalar_kv_dims=16, point_kv_dims=4),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim * 4, dim),
                        ),
                        # Per-layer CA local translation (3D)
                        "coord_update_proj": nn.Linear(dim, 3),
                    }
                )
            )
        # Atom head for per-atom local offsets; reshaped to [B,N,M,3] at runtime
        self.atom_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3 * 32),  # supports up to ~10 atoms; will be cropped/padded
        )

    @staticmethod
    def _residue_frame_from_NCAC(n_pos, ca_pos, c_pos):
        """
        Build local orthonormal frame from N-CA-C:
          x: CA->C; z: normalize((CA->N) × (CA->C)); y: z × x
        Returns rotations [B,N,3,3] and translations [B,N,3] (CA positions)
        """
        v_ca_c = c_pos - ca_pos
        v_ca_n = n_pos - ca_pos
        ex = F.normalize(v_ca_c, dim=-1)
        ez = F.normalize(torch.cross(v_ca_n, v_ca_c, dim=-1), dim=-1)
        ey = torch.cross(ez, ex, dim=-1)
        R = torch.stack([ex, ey, ez], dim=-2)  # [B,N,3,3]
        R = torch.nan_to_num(R)
        t = ca_pos
        return R, t

    def forward(self, seq_features, pair_features, backbone_atoms, residue_mask, atom_mask=None):
        """
        backbone_atoms : [B,N,M,3]  (e.g., M=5 for N/CA/C/O/CB)
        residue_mask   : [B,N]
        atom_mask      : [B,N,M] or None
        Returns: pred_coord_update [B,N,M,3]
        """
        b, n, m, _ = backbone_atoms.shape
        dtype = backbone_atoms.dtype

        # Frame from N/CA/C
        pos_n  = backbone_atoms[:, :, BBHeavyAtom.N,  :]
        pos_ca = backbone_atoms[:, :, BBHeavyAtom.CA, :]
        pos_c  = backbone_atoms[:, :, BBHeavyAtom.C,  :]
        rotations, translations = self._residue_frame_from_NCAC(pos_n, pos_ca, pos_c)
        initial_translations = translations.clone()

        # Iterate layers: only propagate CA translations inside loop
        for layer in self.layers:
            seq_res = seq_features
            normed_seq = layer["norm_seq"](seq_features)
            normed_pair = layer["norm_pair"](pair_features)
            ipa_out = layer["ipa"](normed_seq, normed_pair, rotations, translations, residue_mask)
            seq_features = ipa_out + seq_res
            seq_features = layer["ffn"](seq_features) + seq_features

            ca_update_local = layer["coord_update_proj"](seq_features)            # [B,N,3]
            ca_update_cart  = torch.einsum("b n c, b n r c -> b n r", ca_update_local, rotations)
            translations = translations + ca_update_cart * residue_mask[..., None].to(dtype)

        ca_total_update = translations - initial_translations                     # [B,N,3]

        # Final atom local offsets -> cartesian
        atom_offsets_local_big = self.atom_head(seq_features)                     # [B,N,3*32]
        need = 3 * m
        if atom_offsets_local_big.size(-1) < need:
            atom_offsets_local_big = F.pad(atom_offsets_local_big, (0, need - atom_offsets_local_big.size(-1)))
        atom_offsets_local = atom_offsets_local_big[..., :need].view(b, n, m, 3)  # [B,N,M,3]

        # CA offset is zero; its motion is from accumulated CA translation
        atom_offsets_local[:, :, BBHeavyAtom.CA:BBHeavyAtom.CA+1, :] = 0.0

        atom_offsets_cart = torch.einsum("b n a c, b n r c -> b n a r", atom_offsets_local, rotations)
        pred_pos_atoms = translations[:, :, None, :] + atom_offsets_cart
        pred_coord_update = pred_pos_atoms - backbone_atoms
        pred_coord_update[:, :, BBHeavyAtom.CA, :] = ca_total_update

        if atom_mask is not None:
            valid_atoms = atom_mask.bool() & residue_mask.bool().unsqueeze(-1)
            pred_coord_update = torch.where(
                valid_atoms[..., None],
                pred_coord_update,
                torch.zeros_like(pred_coord_update),
            )
        else:
            pred_coord_update = torch.where(
                residue_mask[..., None, None].bool(),
                pred_coord_update,
                torch.zeros_like(pred_coord_update),
            )
        return pred_coord_update


# ----------------------------
# Main Model
# ----------------------------
class EnhancedRDE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config.get("model", {})
        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.pair_dim = int(model_cfg.get("pair_dim", 128))
        self.n_heads = int(model_cfg.get("n_heads", 8))
        self.dropout = float(model_cfg.get("dropout", 0.1))
        self.num_recycle = int(model_cfg.get("num_recycle", 0))
        self.aux_layer_idx = int(model_cfg.get("aux_layer_idx", 0))
        self.use_chi_hint = bool(model_cfg.get("use_chi_hint", True))
        self.chi_hint_dropout_prob = float(
            model_cfg.get("chi_hint_dropout_prob", 0.0)
        )

        # Embeddings
        self.aa_embed = nn.Embedding(len(AA), self.hidden_dim)
        self.pos_embed = nn.Linear(1, self.pair_dim)
        self.dist_rbf  = RBFExpansion(num_rbfs=32, cutoff=20.0)
        self.dist_proj = nn.Linear(32, self.pair_dim)

        # Node->pair initial mapping
        # (+8 for chi(sin,cos) expanded features replicated to pair)
        self.node_pair_embed = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 8, self.pair_dim), nn.GELU(), nn.Linear(self.pair_dim, self.pair_dim)
        )

        # GVP Geometry encoder
        gnn_cfg = model_cfg.get("gnn", {})
        gnn_in_dim = len(AA) + 3  # onehot + simple physchem
        self.v_dim = int(gnn_cfg.get("v_dim", 8))
        self.geometry_encoder = GVPGeometryEncoder(
            input_dim=gnn_in_dim,
            hidden_dim=self.hidden_dim,
            v_dim=self.v_dim,
            layers=int(gnn_cfg.get("layers", 3)),
            drop=self.dropout,
        )

        # Fusion stack
        self.fusion_stack = nn.ModuleList(
            [FusionLayer(self.hidden_dim, self.pair_dim, self.n_heads, self.dropout) for _ in range(int(model_cfg.get("n_layers", 6)))]
        )

        # IPA structure module(s)
        ipa_cfg = model_cfg.get("ipa", {"depth": 2, "heads": self.n_heads, "dropout": self.dropout})
        self.structure_module = IPAModule(self.hidden_dim, self.pair_dim, int(ipa_cfg.get("depth", 2)), int(ipa_cfg.get("heads", self.n_heads)), float(ipa_cfg.get("dropout", self.dropout)))
        if self.num_recycle > 0 or self.config.get("training", {}).get("aux_loss_weight", 0) > 0:
            self.aux_structure_module = IPAModule(self.hidden_dim, self.pair_dim, int(ipa_cfg.get("depth", 2)), int(ipa_cfg.get("heads", self.n_heads)), float(ipa_cfg.get("dropout", self.dropout)))

        self.output_norm = nn.LayerNorm(self.hidden_dim)

        # χ head (sin/cos)
        self.chi_predictor = self._create_chi_predictor()
        if self.num_recycle > 0 or self.config.get("training", {}).get("aux_loss_weight", 0) > 0:
            self.aux_chi_predictor = self._create_chi_predictor()

        # Simple phys-chem feature buffers
        h_scale = torch.tensor(
            [1.8, 2.5, -3.5, -3.5, 2.8, -3.5, -3.5, 1.8, -1.8, 3.8, 1.6, -3.5, -1.6, -3.5, -1.6, -0.8, -0.7, -0.9, -1.6, 2.5, -0.9, 0.0],
            dtype=torch.float32
        )[:len(AA)]
        mw_scale = torch.tensor(
            [89.1, 174.2, 132.1, 133.1, 121.2, 146.2, 147.1, 131.2, 146.2, 131.2, 131.2, 149.2, 165.2, 115.1, 105.1, 119.1, 240.3, 181.2, 117.1, 101.1, 0.0, 0.0],
            dtype=torch.float32
        )[:len(AA)]
        charge_scale = torch.tensor(
            [0, 1, 0, -1, 0, -1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=torch.float32
        )[:len(AA)]
        self.register_buffer("hydrophobicity", h_scale)
        self.register_buffer("mw_weight", mw_scale)
        self.register_buffer("charge", charge_scale)

    def _create_chi_predictor(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim + 8, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 8),  # 4*(sin,cos)
        )

    def prepare_chi_hint(
        self,
        angles,
        chi_mask=None,
        masked_flag=None,
        use_hint=None,
        apply_dropout=True,
    ):
        """Apply the shared hint token contract and optional train-time dropout."""
        use_hint = self.use_chi_hint if use_hint is None else bool(use_hint)
        encoded = encode_chi_hint(
            angles,
            chi_mask=chi_mask,
            masked_flag=masked_flag,
            use_hint=use_hint,
        )
        if (
            apply_dropout
            and self.training
            and use_hint
            and self.chi_hint_dropout_prob > 0
        ):
            drop_residue = torch.rand(
                angles.shape[:-1], device=angles.device
            ) < self.chi_hint_dropout_prob
            encoded = torch.where(
                drop_residue.unsqueeze(-1), torch.zeros_like(encoded), encoded
            )
        return encoded

    def _get_predictions(self, context, pair_rep, chi_corrupt_sincos, backbone_atoms, atom_mask, mask, chi_head, coord_head):
        # χ prediction
        predictor_input = torch.cat([context, chi_corrupt_sincos], dim=-1)
        pred_chi_sincos = chi_head(predictor_input)
        pred_chi_sincos = rearrange(pred_chi_sincos, "b n (c s) -> b n c s", c=4, s=2)
        pred_chi_sincos = pred_chi_sincos / (pred_chi_sincos.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        pred_chi_sincos = torch.where(
            mask[..., None, None].bool(),
            pred_chi_sincos,
            torch.zeros_like(pred_chi_sincos),
        )

        # Multi-atom coordinate updates
        pred_coord_update = coord_head(context, pair_rep, backbone_atoms, mask, atom_mask)
        return pred_chi_sincos, pred_coord_update

    @staticmethod
    def _get_backbone_local_frames(pos_heavyatom: torch.Tensor) -> torch.Tensor:
        """
        Builds a local orthonormal basis for each residue using N, CA, C.
        """
        pos_n = pos_heavyatom[:, :, BBHeavyAtom.N, :]
        pos_ca = pos_heavyatom[:, :, BBHeavyAtom.CA, :]
        pos_c = pos_heavyatom[:, :, BBHeavyAtom.C, :]

        v_ca_c = pos_c - pos_ca
        v_ca_n = pos_n - pos_ca
        eps = 1e-6

        basis_x = F.normalize(v_ca_c, dim=-1, p=2, eps=eps)
        basis_z = F.normalize(torch.cross(v_ca_n, v_ca_c, dim=-1), dim=-1, p=2, eps=eps)
        basis_y = torch.cross(basis_z, basis_x, dim=-1)

        local_frames = torch.stack([basis_x, basis_y, basis_z], dim=2)
        if not torch.isfinite(local_frames).all():
            local_frames = torch.nan_to_num(local_frames)
        return local_frames

    @staticmethod
    def _build_graph(X: torch.Tensor, mask: torch.Tensor, use_radius=False, radius=12.0, k=16):
        B, L, _ = X.shape
        edges = []
        for b in range(B):
            valid = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if valid.numel() <= 1:
                continue
            pos = X[b, valid]
            e = None
            if use_radius:
                e = radius_graph(pos, r=float(radius), loop=False)
            e_knn = knn_graph(pos, k=min(int(k), max(pos.size(0)-1, 1)), loop=False)
            e = e_knn if e is None else torch.unique(torch.cat([e, e_knn], dim=1), dim=1)
            e = valid[e]
            if b > 0:
                e = e + b * L
            edges.append(e)
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=X.device)
        return torch.cat(edges, dim=1) if len(edges) > 1 else edges[0]

    def forward(self, batch: Dict[str, torch.Tensor]):
        aa = batch["aa"]
        mask = batch.get("mask")
        if mask is None:
            atom_mask = batch.get("mask_atoms", batch.get("mask_heavyatom"))
            if atom_mask is not None:
                mask = atom_mask.bool().any(dim=-1)
            else:
                mask = torch.ones_like(aa, dtype=torch.bool)
        mask = mask.bool()
        b, n = aa.shape
        pos_heavyatom = batch["pos_heavyatom"]
        backbone_atoms = batch.get("pos_atoms", pos_heavyatom)
        pos_alpha = backbone_atoms[:, :, BBHeavyAtom.CA, :]

        # Local frames for GVP initial vectors
        local_frames = self._get_backbone_local_frames(backbone_atoms)
        v_init = torch.zeros(b, n, self.v_dim, 3, device=aa.device, dtype=pos_alpha.dtype)
        num_vectors_to_use = min(3, self.v_dim)
        v_init[:, :, :num_vectors_to_use, :] = local_frames[:, :, :num_vectors_to_use, :]
        v_init = torch.where(
            mask[..., None, None], v_init, torch.zeros_like(v_init)
        )

        # Use full atom block for IPA; fall back to pos_heavyatom
        atom_mask = batch.get("mask_atoms", batch.get("mask_heavyatom", None))  # [B,N,M] or None

        # Sequence & simple phys-chem
        aa_onehot = F.one_hot(aa, num_classes=len(AA)).float()
        h_feats = torch.stack([
            self.hydrophobicity[aa],
            self.mw_weight[aa],
            self.charge[aa]
        ], dim=-1)

        seq_embed = self.aa_embed(aa)
        seq_context = seq_embed

        # Pair features
        rel_pos = pos_alpha[:, :, None, :] - pos_alpha[:, None, :, :]
        rel_dist = torch.norm(rel_pos, p=2, dim=-1, keepdim=True)

        chi_corrupt = batch.get(
            "chi_corrupt",
            torch.zeros(b, n, 4, device=aa.device, dtype=seq_embed.dtype),
        )
        chi_corrupt_sincos = self.prepare_chi_hint(
            chi_corrupt,
            chi_mask=batch.get("chi_mask"),
            masked_flag=batch.get("chi_masked_flag"),
        )

        node_pair_feat = torch.cat([
            repeat(seq_context, "b i d -> b i j d", j=n),
            repeat(seq_context, "b j d -> b i j d", i=n),
            repeat(chi_corrupt_sincos, "b j d -> b i j d", i=n),
        ], dim=-1)
        rbf = self.dist_rbf(rel_dist.squeeze(-1))
        dist_emb = self.dist_proj(rbf)
        pair_mask = (mask[:, :, None] & mask[:, None, :]).to(dist_emb.dtype)
        pair_rep_init_raw = self.node_pair_embed(node_pair_feat) + dist_emb
        pair_rep_init = torch.where(
            pair_mask[..., None].bool(),
            pair_rep_init_raw,
            torch.zeros_like(pair_rep_init_raw),
        )

        prev_pair_rep = torch.zeros_like(pair_rep_init)
        prev_seq_context = torch.zeros_like(seq_context)
        aux_preds = None

        # GVP geometry encoding
        geometry_feat = torch.cat([aa_onehot, h_feats], dim=-1)
        geo_feat_all = geometry_feat.view(b * n, -1)
        x_all = pos_alpha.view(b * n, 3)
        v_init_all = v_init.view(b * n, self.v_dim, 3)

        gnn_cfg = self.config.get("model", {}).get("gnn", {})
        edge_index = self._build_graph(pos_alpha, mask, use_radius=gnn_cfg.get("use_radius", False),
                                       radius=gnn_cfg.get("radius", 12.0), k=gnn_cfg.get("k_neighbors", 16))

        geometry_context_all = self.geometry_encoder(geo_feat_all, x_all, edge_index, v_init=v_init_all)
        geometry_context = geometry_context_all.view(b, n, self.hidden_dim)
        geometry_context = torch.where(
            mask[..., None], geometry_context, torch.zeros_like(geometry_context)
        )

        # Recycling loop
        training_cfg = self.config.get("training", {})
        for cycle in range(self.num_recycle + 1):
            seq_context_recycled = seq_context + prev_seq_context.detach()
            pair_rep_recycled = pair_rep_init + prev_pair_rep.detach()

            for layer_idx, layer in enumerate(self.fusion_stack):
                seq_context_recycled, pair_rep_recycled = layer(seq_context_recycled, geometry_context, pair_rep_recycled, mask)

                is_last_cycle = (cycle == self.num_recycle)
                use_aux_heads = training_cfg.get("aux_loss_weight", 0) > 0
                if is_last_cycle and use_aux_heads and (layer_idx == int(self.aux_layer_idx)) and self.training:
                    aux_context = self.output_norm(seq_context_recycled)
                    aux_pair_rep = pair_rep_recycled
                    aux_preds = self._get_predictions(aux_context, aux_pair_rep, chi_corrupt_sincos,
                                                      backbone_atoms, atom_mask, mask,
                                                      self.aux_chi_predictor, self.aux_structure_module)

            prev_pair_rep = pair_rep_recycled
            prev_seq_context = seq_context_recycled

        final_context = self.output_norm(seq_context_recycled)
        final_pair_rep = prev_pair_rep
        final_preds = self._get_predictions(final_context, final_pair_rep, chi_corrupt_sincos,
                                            backbone_atoms, atom_mask, mask,
                                            self.chi_predictor, self.structure_module)

        return final_preds, aux_preds

