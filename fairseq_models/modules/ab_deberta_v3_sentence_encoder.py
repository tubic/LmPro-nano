# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.data import Dictionary

from .ab_DeBertaV3EncoderLayer import DeBertaV3EncoderLayer

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
class AntibodyDeBertaV3SentenceEncoder(nn.Module):
    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 512,
        segments_vocab: Dictionary = None,
        encoder_normalize_before: bool = True,
        apply_bert_init: bool = False,
        activation_fn: str = "gelu",
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        # DeBERTa v3 specific
        max_relative_positions: int = -1,
        pos_att_type: str = "c2p|p2c",
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = len(segments_vocab)
        self.apply_bert_init = apply_bert_init
        self.max_relative_positions = max_relative_positions
        self.pos_att_type = pos_att_type
        self.layer_norm_eps = layer_norm_eps

        self.embed_tokens = self.build_embedding(self.vocab_size, self.embedding_dim, self.padding_idx)

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=segments_vocab.pad())
            if self.num_segments > 0
            else None
        )

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_deberta_v3_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    max_relative_positions=max_relative_positions,
                    pos_att_type=pos_att_type,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, eps=self.layer_norm_eps)
        else:
            self.emb_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

    def build_deberta_v3_encoder_layer(
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
        return DeBertaV3EncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            max_relative_positions=max_relative_positions,
            pos_att_type=pos_att_type,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask
        padding_mask = tokens.eq(self.padding_idx)
        if not self.traceable and not padding_mask.any():
            padding_mask = None

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.embed_tokens(tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        x = x.transpose(0, 1)  # B x T x C -> T x B x C

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(x, padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        return inner_states, sentence_rep