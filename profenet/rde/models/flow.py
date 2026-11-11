# rde/models/flow.py
import torch
import torch.nn as nn
import numpy as np
from .flows.container import ContextualSequentialFlow
from .flows.coupling import ContextualCircularSplineCouplingLayer

class NormalizingFlowHead(nn.Module):
    def __init__(self, data_dim, cond_dim, hidden_dim, num_bins, num_layers, num_blocks):
        super().__init__()
        self.data_dim = data_dim
        
        # 构建交替的耦合层序列
        flows = []
        for i in range(num_blocks):
            # 交替变换前一半和后一半的维度
            if i % 2 == 0:
                mapping_dims = list(range(data_dim // 2))
            else:
                mapping_dims = list(range(data_dim // 2, data_dim))

            flows.append(
                ContextualCircularSplineCouplingLayer(
                    n_dims=data_dim,
                    mapping_dims=mapping_dims,
                    n_context_dims=cond_dim,
                    n_hidden_dims=hidden_dim,
                    n_spline_bins=num_bins,
                )
            )
        
        self.flow = ContextualSequentialFlow(flows)
        
        # 为基础分布注册参数
        self.register_buffer('base_dist_low', torch.tensor(0.0))
        self.register_buffer('base_dist_high', torch.tensor(2 * np.pi))

    def log_prob(self, x, cond):
        if x.numel() == 0:
            return torch.zeros(0, device=x.device)
        
        # 变换到基础分布
        y, logabsdet = self.flow(x, cond, inverse=False)
        
        # 计算基础分布的对数概率
        base_dist = torch.distributions.Uniform(self.base_dist_low, self.base_dist_high)
        log_prob_base = base_dist.log_prob(y).sum(dim=-1)
        
        # 根据变量替换公式，返回最终的对数概率
        return log_prob_base + logabsdet

    def sample(self, num_samples, cond):
        # 从基础分布中采样
        base_dist = torch.distributions.Uniform(self.base_dist_low, self.base_dist_high)
        y = base_dist.sample((num_samples, self.data_dim)).to(cond.device)

        # 通过流的反向过程进行变换，得到目标分布的样本
        x, _ = self.flow(y, cond, inverse=True)
        return x
