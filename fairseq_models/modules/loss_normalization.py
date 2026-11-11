"""Loss contracts shared by the antibody generation decoders and criterion.

Fairseq divides accumulated gradients by the ``sample_size`` returned by a
criterion.  The historical ABGNN implementation returned already-averaged
losses together with a token sample size, which made both the update magnitude
and the structure/sequence balance depend on micro-batch size.  The
``token_mean`` mode below follows the standard Fairseq contract: return a
token-summed loss and let the trainer perform the single division.
"""

import torch


LOSS_NORMALIZATIONS = ("legacy", "token_mean")


def masked_mean_per_example(values, mask):
    """Average a masked geometry tensor independently for each example."""
    mask = mask.to(dtype=values.dtype, device=values.device)
    reduce_dims = tuple(range(1, values.dim()))
    if not reduce_dims:
        return values
    numerator = (values * mask).sum(dim=reduce_dims)
    denominator = mask.sum(dim=reduce_dims).clamp_min(1)
    return numerator / denominator


def token_weighted_structure_mean(
    loss_mask_pairs, cdr_token_counts, refine_iterations
):
    """Combine per-example geometry means using CDR-token weights.

    This makes the structure objective additive across gradient-accumulation
    micro-batches once ``combine_generation_loss`` converts it to a token sum.
    """
    per_example = None
    for values, mask in loss_mask_pairs:
        term = masked_mean_per_example(values, mask)
        per_example = term if per_example is None else per_example + term
    if per_example is None:
        raise ValueError("loss_mask_pairs must contain at least one term")

    weights = cdr_token_counts.to(
        dtype=per_example.dtype, device=per_example.device
    )
    weighted_mean = (per_example * weights).sum() / weights.sum().clamp_min(1)
    return weighted_mean / refine_iterations


def _positive(value):
    """Avoid a malformed empty CDR batch producing a division by zero."""
    if torch.is_tensor(value):
        return value.clamp_min(1)
    return max(value, 1)


def _validate(mode):
    if mode not in LOSS_NORMALIZATIONS:
        raise ValueError(
            "loss_normalization must be one of {} (got {!r})".format(
                LOSS_NORMALIZATIONS, mode
            )
        )


def normalize_refinement_losses(
    sequence_sum,
    structure_sum,
    cdr_token_count,
    refine_iterations,
    mode,
    legacy_structure_denominator=None,
    legacy_average_refinement=True,
):
    """Return per-token sequence and per-example structure loss means.

    ``structure_sum`` is formed from geometry terms that have already been
    averaged over their own valid coordinate/pair masks.  Dividing it once by
    refinement iterations yields a batch-size-stable geometric loss.  The
    legacy branch retains each decoder's historical normalization, including
    the stack ablation's missing refinement average, so existing reproduction
    runs remain numerically compatible.
    """
    _validate(mode)
    token_count = _positive(cdr_token_count)
    sequence_mean = sequence_sum / token_count / refine_iterations
    structure_mean = structure_sum / refine_iterations

    if mode == "legacy":
        if not legacy_average_refinement:
            sequence_mean = sequence_sum / token_count
            structure_mean = structure_sum
        denominator = (
            token_count
            if legacy_structure_denominator is None
            else _positive(legacy_structure_denominator)
        )
        structure_mean = structure_mean / denominator

    return sequence_mean, structure_mean


def combine_generation_loss(
    encoder_sum,
    decoder_sequence_mean,
    decoder_structure_mean,
    sample_size,
    encoder_scale,
    decoder_sequence_scale,
    decoder_structure_scale,
    mode,
):
    """Combine loss terms using either the legacy or Fairseq-correct contract.

    Returns ``(loss_for_backward, encoder_mean)``.  In ``token_mean`` mode the
    first value is a token sum, so Fairseq's later division by ``sample_size``
    produces a true weighted mean independent of micro-batch partitioning.
    """
    _validate(mode)
    token_count = _positive(sample_size)
    encoder_mean = encoder_sum / token_count

    if mode == "legacy":
        loss = (
            encoder_scale * encoder_mean
            + decoder_sequence_scale * decoder_sequence_mean
            + decoder_structure_scale * decoder_structure_mean
        )
    else:
        loss = (
            encoder_scale * encoder_sum
            + token_count
            * (
                decoder_sequence_scale * decoder_sequence_mean
                + decoder_structure_scale * decoder_structure_mean
            )
        )

    return loss, encoder_mean
