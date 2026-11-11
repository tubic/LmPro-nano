# rde/models/rde_ddg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
import logging
from torch_geometric.nn import GATv2Conv, knn_graph

from rde.utils.protein.constants import BBHeavyAtom, AA

logger = logging.getLogger(__name__)

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gnn_layers.append(GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=True, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))
        self.output_proj = nn.Linear(hidden_dim * heads, hidden_dim)

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        for i, (gnn_layer, norm_layer) in enumerate(zip(self.gnn_layers, self.norms)):
            x_res = x
            x_update = gnn_layer(x, edge_index)
            if i > 0 and x_res.shape == x_update.shape:
                x = x_res + x_update
            else:
                 x = x_update
            x = norm_layer(x)
        return self.output_proj(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
    
    def forward(self, query, context):
        attn_output, _ = self.attention(query, context, context)
        query = self.layer_norm1(query + attn_output)
        ffn_output = self.ffn(query)
        query = self.layer_norm2(query + ffn_output)
        return query

class FusionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.geo_fusion = CrossAttentionFusion(hidden_dim, n_heads, dropout)
        self.topo_fusion = CrossAttentionFusion(hidden_dim, n_heads, dropout)
        self.dyn_fusion = CrossAttentionFusion(hidden_dim, n_heads, dropout)
        self.ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
    
    def forward(self, seq_context, geo_context, topo_context, dyn_context):
        seq_context = self.geo_fusion(seq_context, geo_context)
        seq_context = self.topo_fusion(seq_context, topo_context)
        seq_context = self.dyn_fusion(seq_context, dyn_context)
        seq_context = seq_context + self.ffn(seq_context)
        return seq_context

class EnhancedRDE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config['model']
        feature_cfg = config['features']
        gnn_cfg = model_cfg['gnn']
        fusion_cfg = model_cfg['fusion']
        self.hidden_dim = model_cfg['hidden_dim']
        self.embedding = nn.Embedding(21, model_cfg['input_dim'])
        self.encoder = nn.Linear(model_cfg['input_dim'] + 3, self.hidden_dim)
        geo_input_dim = 3 + len(feature_cfg['geometry']['hks']['timesteps'])
        self.geo_gnn_encoder = GNNEncoder(geo_input_dim, self.hidden_dim, gnn_cfg['num_gnn_layers'], gnn_cfg['dropout'])
        topo_input_dim = 15
        self.topo_gnn_encoder = GNNEncoder(topo_input_dim, self.hidden_dim, gnn_cfg['num_gnn_layers'], gnn_cfg['dropout'])
        dyn_input_dim = 1 + feature_cfg['dynamics']['n_low_freq_modes_to_keep']
        self.dyn_gnn_encoder = GNNEncoder(dyn_input_dim, self.hidden_dim, gnn_cfg['num_gnn_layers'], gnn_cfg['dropout'])
        self.fusion_stack = nn.ModuleList([FusionLayer(self.hidden_dim, fusion_cfg['attention_heads'], fusion_cfg['dropout']) for _ in range(fusion_cfg['num_fusion_layers'])])
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_layer = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, 4 * 2))
        h_scale = torch.tensor([1.8, 2.5, -3.5, -3.5, 2.8, -3.5, -0.4, 4.5, -3.2, 4.2, 3.8, 1.9, -1.3, -4.5, -0.8, -0.7, -0.9, -1.6, 2.5, -0.9, 0.0], dtype=torch.float32)
        mw_scale = torch.tensor([89.1, 174.2, 132.1, 133.1, 121.2, 147.1, 146.2, 75.1, 155.2, 204.2, 131.2, 149.2, 165.2, 117.1, 105.1, 119.1, 240.3, 181.2, 117.1, 101.1, 0.0], dtype=torch.float32)
        charge_scale = torch.tensor([0, 1, 0, -1, 0, -1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        self.register_buffer('hydrophobicity', h_scale)
        self.register_buffer('molecular_weight', mw_scale)
        self.register_buffer('charge', charge_scale)

    def forward(self, batch, return_features=False):
        aa = batch['aa']
        mask = batch['mask']
        batch_size, n_nodes = aa.shape
        pos_heavyatom_tensor = batch['pos_heavyatom']
        pos_alpha_tensor = pos_heavyatom_tensor[:, :, BBHeavyAtom.CA, :]
        batch_idx = torch.arange(batch_size, device=aa.device).repeat_interleave(n_nodes)
        alpha_coords_flat = pos_alpha_tensor.reshape(-1, 3)
        active_nodes_mask = mask.reshape(-1)
        coords_for_graph = alpha_coords_flat[active_nodes_mask]
        batch_for_graph = batch_idx[active_nodes_mask]
        if coords_for_graph.shape[0] == 0:
            return torch.zeros(batch_size, n_nodes, self.hidden_dim if return_features else 4*2, device=aa.device)
        edge_index = knn_graph(coords_for_graph, k=self.config['model']['gnn']['k_neighbors'], batch=batch_for_graph)
        base_feat = self.embedding(aa)
        phys_feat = torch.stack([self.hydrophobicity[aa], self.molecular_weight[aa], self.charge[aa]], dim=-1)
        seq_context = self.encoder(torch.cat([base_feat, phys_feat], dim=-1))
        def get_context(batch_feat, encoder):
            feat_flat = batch_feat.reshape(-1, batch_feat.shape[-1])[active_nodes_mask]
            context_flat = encoder(feat_flat, edge_index)
            context = torch.zeros(batch_size, n_nodes, self.hidden_dim, device=seq_context.device)
            context[mask] = context_flat
            return context
        geo_context = get_context(batch['geometric_features'], self.geo_gnn_encoder)
        topo_context = get_context(batch['topological_features'], self.topo_gnn_encoder)
        dyn_context = get_context(batch['dynamics_features'], self.dyn_gnn_encoder)
        for layer in self.fusion_stack:
            seq_context = layer(seq_context, geo_context, topo_context, dyn_context)
        final_representation = self.output_norm(seq_context)
        if return_features:
            return final_representation
        output_flat = self.output_layer(final_representation)
        return output_flat.view(batch_size, n_nodes, 4, 2)

class DDG_RDE_Network(nn.Module):
    def __init__(self, cfg, finetune_mode=False, **kwargs):
        super().__init__()
        self.cfg = cfg
        if not finetune_mode:
            raise ValueError("DDG_RDE_Network must be run in finetune_mode.")
        self.rde_feature_extractor = EnhancedRDE(self.cfg.rde)
        logger.info(f"Loading pretrained RDE weights from: {self.cfg.rde.checkpoint}")
        ckpt = torch.load(self.cfg.rde.checkpoint, map_location='cpu')
        self.rde_feature_extractor.load_state_dict(ckpt.get('model_state_dict', ckpt))
        for param in self.rde_feature_extractor.parameters():
            param.requires_grad = False
        self.rde_feature_extractor.eval()
        logger.info("Froze all parameters of the RDE feature extractor.")
        feature_dim = self.cfg.rde.model.hidden_dim 
        self.ddg_readout = nn.Sequential(
            nn.Linear(feature_dim * 2, self.cfg.ddg.hidden_dim), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(self.cfg.ddg.hidden_dim, self.cfg.ddg.hidden_dim // 2), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(self.cfg.ddg.hidden_dim // 2, 1)
        )

    def _get_structure_features(self, data_dict):
        self.rde_feature_extractor.eval()
        with torch.no_grad():
            return self.rde_feature_extractor(data_dict, return_features=True)

    def forward(self, data):
        data = EasyDict(data)
        wt_batch = {k.replace('_wt', ''): v for k, v in data.items() if k.endswith('_wt')}
        mut_batch = {k.replace('_mut', ''): v for k, v in data.items() if k.endswith('_mut')}
        shared_keys = ['mask', 'mut_flag', 'pos_heavyatom']
        for key in shared_keys:
            if key in data:
                wt_batch[key] = data[key]
                mut_batch[key] = data[key]
        feature_keys = ['geometric_features', 'topological_features', 'dynamics_features']
        for key in feature_keys:
            wt_batch[key] = data[f'{key}_wt']
            mut_batch[key] = data[f'{key}_mut']
        h_wt = self._get_structure_features(wt_batch)
        h_mut = self._get_structure_features(mut_batch)
        mut_flag = data.mut_flag.unsqueeze(-1)
        h_wt_mut = (h_wt * mut_flag).sum(dim=1)
        h_mut_mut = (h_mut * mut_flag).sum(dim=1)
        h_cat = torch.cat([h_wt_mut, h_mut_mut], dim=-1)
        ddg_pred = self.ddg_readout(h_cat).squeeze(-1)
        loss = F.mse_loss(ddg_pred, data.get('ddG', data.get('ddg')).view_as(ddg_pred))
        return {'regression': loss}, ddg_pred

def model_factory(cfg, **kwargs):
    return DDG_RDE_Network(cfg.model, **kwargs)