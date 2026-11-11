"""Frozen Lm-Nano encoder interface for external representation benchmarks."""

import os
import sys

import torch
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.data import Dictionary


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fairseq_models.models.ab_roberta_base import AntibodyRobertaModel  # noqa: E402


class LmNanoEncoder(torch.nn.Module):
    """Load the pretraining checkpoint without its optimizer or downstream decoder."""

    def __init__(self, checkpoint, device="cpu", unknown_policy="error"):
        super().__init__()
        if unknown_policy not in {"error", "unk"}:
            raise ValueError("unknown_policy must be 'error' or 'unk'")
        state = load_checkpoint_to_cpu(checkpoint, arg_overrides=None)
        args = state["args"]
        if getattr(args, "finetune", False):
            raise ValueError("NbBench requires a pretraining encoder checkpoint, not finetuning")

        seq_dictionary = Dictionary.load(os.path.join(ROOT, "data", "seq_dict.txt"))
        tag_dictionary = Dictionary.load(os.path.join(ROOT, "data", "tag_dict.txt"))
        mask_index = seq_dictionary.add_symbol("<mask>")

        class TaskConfig:
            source_dictionary = seq_dictionary
            tag_source_dictionary = tag_dictionary

        model = AntibodyRobertaModel.build_model(args, TaskConfig())
        model.load_state_dict(state["model"], strict=True, args=args)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        self.model = model.to(device)
        self.args = args
        self.seq_dictionary = seq_dictionary
        self.tag_dictionary = tag_dictionary
        self.mask_index = mask_index
        self.framework_tag = tag_dictionary.index("0")
        self.max_residues = int(args.max_positions) - 1  # reserve one position for BOS/CLS
        self.checkpoint = os.path.abspath(checkpoint)
        self.unknown_policy = unknown_policy
        self.unknown_residue_count = 0

    @property
    def device(self):
        return next(self.model.parameters()).device

    def tokenize(self, sequences):
        lengths = [len(sequence) for sequence in sequences]
        if not lengths:
            raise ValueError("empty sequence batch")
        if max(lengths) > self.max_residues:
            raise ValueError(
                f"sequence length {max(lengths)} exceeds Lm-Nano limit {self.max_residues}"
            )

        width = max(lengths) + 1
        tokens = torch.full(
            (len(sequences), width), self.seq_dictionary.pad(), dtype=torch.long
        )
        tags = torch.full(
            (len(sequences), width), self.tag_dictionary.pad(), dtype=torch.long
        )
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            tokens[row, 0] = self.seq_dictionary.bos()
            tags[row, 0] = self.tag_dictionary.bos()
            attention_mask[row, : len(sequence) + 1] = True
            for column, residue in enumerate(sequence, start=1):
                if residue == "-":
                    token = self.mask_index
                else:
                    token = self.seq_dictionary.index(residue)
                    if token == self.seq_dictionary.unk():
                        if self.unknown_policy == "error":
                            raise ValueError(
                                f"unsupported residue {residue!r} in sequence {row}"
                            )
                        self.unknown_residue_count += 1
                tokens[row, column] = token
                # Ground-truth CDR tags are deliberately hidden from every NbBench task.
                tags[row, column] = self.framework_tag
        return tokens.to(self.device), tags.to(self.device), attention_mask.to(self.device)

    @torch.no_grad()
    def forward(self, sequences):
        tokens, tags, attention_mask = self.tokenize(sequences)
        features, _ = self.model(
            src_tokens=tokens,
            tag_tokens=tags,
            features_only=True,
            return_all_hiddens=False,
        )
        return features, attention_mask

    @torch.no_grad()
    def cls_embeddings(self, sequences):
        features, _ = self.forward(sequences)
        return features[:, 0]

    @torch.no_grad()
    def mean_embeddings(self, sequences):
        features, attention_mask = self.residue_embeddings(sequences)
        weights = attention_mask.unsqueeze(-1).to(features.dtype)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    @torch.no_grad()
    def residue_embeddings(self, sequences):
        features, attention_mask = self.forward(sequences)
        return features[:, 1:], attention_mask[:, 1:]
