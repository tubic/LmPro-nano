# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("antibody_masked_lm")
class AntibodyMaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample["target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        logits = model(
            src_tokens=sample["net_input0"]["src_tokens"],
            tag_tokens=sample["net_input1"]["src_tokens"],
            masked_tokens=masked_tokens,
        )[0]
        targets = model.get_targets(sample, [logits])
        targets = targets[masked_tokens]

        token_losses = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
            ignore_index=self.padding_idx,
        )
        loss = token_losses.sum()

        predictions = logits.view(-1, logits.size(-1)).argmax(dim=-1)
        correct = predictions.eq(targets.view(-1))
        mask_mode = sample.get("mask_mode")
        if mask_mode is None:
            mask_mode = torch.zeros(
                masked_tokens.size(0), dtype=torch.long, device=masked_tokens.device
            )
        full_rows = mask_mode.to(masked_tokens.device).bool().view(-1)
        token_is_full = full_rows.unsqueeze(1).expand_as(masked_tokens)[masked_tokens]
        token_is_partial = ~token_is_full

        full_correct_matrix = torch.ones_like(masked_tokens, dtype=torch.bool)
        full_correct_matrix[masked_tokens] = correct
        full_has_targets = masked_tokens.any(dim=1)
        full_exact = (
            full_correct_matrix.all(dim=1) & full_rows & full_has_targets
        ).sum()
        logging_token_losses = token_losses if self.tpu else token_losses.detach()

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "ncorrect": correct.sum(),
            "full_cdr3_loss": logging_token_losses[token_is_full].sum(),
            "full_cdr3_sample_size": token_is_full.sum(),
            "full_cdr3_ncorrect": correct[token_is_full].sum(),
            "full_cdr3_nsequences": (full_rows & full_has_targets).sum(),
            "full_cdr3_nexact": full_exact,
            "partial_loss": logging_token_losses[token_is_partial].sum(),
            "partial_sample_size": token_is_partial.sum(),
            "partial_ncorrect": correct[token_is_partial].sum(),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        if sample_size > 0:
            metrics.log_scalar(
                "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=2
            )

        AntibodyMaskedLmLoss._reduce_mask_mode_metrics(
            logging_outputs, "full_cdr3"
        )
        AntibodyMaskedLmLoss._reduce_mask_mode_metrics(
            logging_outputs, "partial"
        )

        full_nsequences = sum(
            log.get("full_cdr3_nsequences", 0) for log in logging_outputs
        )
        if full_nsequences > 0:
            full_nexact = sum(
                log.get("full_cdr3_nexact", 0) for log in logging_outputs
            )
            metrics.log_scalar(
                "full_cdr3_exact_accuracy",
                100.0 * full_nexact / full_nsequences,
                full_nsequences,
                round=2,
            )

    @staticmethod
    def _reduce_mask_mode_metrics(logging_outputs, prefix):
        sample_size = sum(
            log.get(prefix + "_sample_size", 0) for log in logging_outputs
        )
        if sample_size <= 0:
            return
        loss_sum = sum(log.get(prefix + "_loss", 0) for log in logging_outputs)
        ncorrect = sum(
            log.get(prefix + "_ncorrect", 0) for log in logging_outputs
        )
        loss_name = prefix + "_loss"
        metrics.log_scalar(
            loss_name,
            loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_derived(
            prefix + "_ppl",
            lambda meters, name=loss_name: utils.get_perplexity(meters[name].avg),
        )
        metrics.log_scalar(
            prefix + "_accuracy",
            100.0 * ncorrect / sample_size,
            sample_size,
            round=2,
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
