# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from .ab_multihead_attention import MultiheadAttention

class DeBertaV3EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        max_relative_positions,
        pos_att_type,
        layer_norm_eps,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.activation_dropout_module = FairseqDropout(activation_dropout, module_name=self.__class__.__name__)
        
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            max_relative_positions=max_relative_positions,
            pos_att_type=pos_att_type,
        )
        
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, eps=layer_norm_eps)
        
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        
        self.final_layer_norm = LayerNorm(self.embedding_dim, eps=layer_norm_eps)
        
    def forward(
        self,
        x,
        padding_mask=None,
    ):
        residual = x
        x, attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn_weights