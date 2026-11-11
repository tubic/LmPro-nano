"""
The code is adapted from: https://github.com/bayesiains/nsf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0., right=1., bottom=0., top=1.,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError('Inputs must be in [{}, {}], got [{}, {}]'.format(
            left, right,
            torch.min(inputs).item(), torch.max(inputs).item()
        ))

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths.clamp(min=1e-8)
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives
                                             + input_derivatives_plus_one
                                             - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
                                              + input_derivatives_plus_one
                                              - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        discriminant = F.relu(discriminant)

        root = (2 * c) / (-b - torch.sqrt(discriminant) + 1e-8)
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator.clamp(min=1e-8)) - 2 * torch.log(denominator.clamp(min=1e-8))

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths.clamp(min=1e-8)
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator.clamp(min=1e-8)

        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator.clamp(min=1e-8)) - 2 * torch.log(denominator.clamp(min=1e-8))

        return outputs, logabsdet


def circular_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
):
    right_boundary_flag = (inputs == 2*np.pi)
    inputs = inputs % (2*np.pi)
    inputs = torch.where(right_boundary_flag, inputs + 2*np.pi, inputs)
    
    assert unnormalized_derivatives.size(-1) == unnormalized_widths.size(-1)
    unnormalized_derivatives = torch.cat(
        [
            unnormalized_derivatives, 
            unnormalized_derivatives[..., :1]
        ],
        dim=-1,
    )

    outputs, logabsdet = rational_quadratic_spline(
        inputs = inputs,
        unnormalized_widths = unnormalized_widths,
        unnormalized_heights = unnormalized_heights,
        unnormalized_derivatives = unnormalized_derivatives,
        inverse = inverse,
        left = 0.0, right = 2*np.pi,
        bottom = 0.0, top = 2*np.pi,
    )
    outputs = outputs % (2*np.pi)
    return outputs, logabsdet


class ContextualCircularSplineFlow(nn.Module):

    def __init__(self, n_context_dims, n_hidden_dims, n_spline_bins):
        super().__init__()
        self.n_context_dims = n_context_dims
        self.n_hidden_dims = n_hidden_dims
        self.n_spline_bins = n_spline_bins
        self.condition_net = nn.Sequential(
            nn.Linear(n_context_dims, n_hidden_dims), nn.LeakyReLU(),
            nn.Linear(n_hidden_dims, n_hidden_dims), nn.LeakyReLU(),
            nn.Linear(n_hidden_dims, n_spline_bins*3)
        )

    def _get_spline_params(self, c):
        outs = self.condition_net(c)
        outs = outs.unsqueeze(-2)
        w, h, d = torch.split(outs, [self.n_spline_bins]*3, dim=-1)
        return w, h, d

    def forward(self, x, c, inverse):
        assert x.size(-1) == 1
        w, h, d = self._get_spline_params(c)
        y, logabsdet = circular_quadratic_spline(
            inputs = x,
            unnormalized_widths = w,
            unnormalized_heights = h,
            unnormalized_derivatives = d,
            inverse = inverse,
        )
        logabsdet = logabsdet.sum(dim=-1)
        return y, logabsdet
