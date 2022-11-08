import torch
import torch.nn.functional
from functools import reduce
from operator import mul
import sys

sys.path.append("../")
sys.path.append("./")
from utils.util import dsnt, flat_softmax, linear_expectation, normalized_linspace


def euclidean_losses(actual, target):
    assert actual.size() == target.size(
    ), 'input tensors must have the same size'
    return torch.norm(actual - target, p=2, dim=-1, keepdim=False)


def make_gauss(means, size, sigma, normalize=True):

    dim_range = range(-1, -(len(size) + 1), -1)
    coords_list = [
        normalized_linspace(s, dtype=means.dtype, device=means.device)
        for s in reversed(size)
    ]

    # PDF = exp(-(x - \mu)^2 / (2 \sigma^2))

    # dists <- (x - \mu)^2
    dists = [(x - mean)**2 for x, mean in zip(coords_list, means.split(1, -1))]

    # ks <- -1 / (2 \sigma^2)
    stddevs = [2 * sigma / s for s in reversed(size)]
    ks = [-0.5 * (1 / stddev)**2 for stddev in stddevs]

    exps = [(dist * k).exp() for k, dist in zip(ks, dists)]

    # Combine dimensions of the Gaussian
    gauss = reduce(mul, [
        reduce(lambda t, d: t.unsqueeze(d),
               filter(lambda d: d != dim, dim_range), dist)
        for dim, dist in zip(dim_range, exps)
    ])

    if not normalize:
        return gauss

    # Normalize the Gaussians
    val_sum = reduce(lambda t, dim: t.sum(dim, keepdim=True), dim_range,
                     gauss) + 1e-24
    return gauss / val_sum


def _kl(p, q, ndims):
    eps = 1e-24
    unsummed_kl = p * ((p + eps).log() - (q + eps).log())
    kl_values = reduce(lambda t, _: t.sum(-1, keepdim=False), range(ndims),
                       unsummed_kl)
    return kl_values


def _js(p, q, ndims):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, ndims) + 0.5 * _kl(q, m, ndims)


def _divergence_reg_losses(heatmaps, mu_t, sigma_t, divergence):
    ndims = mu_t.size(-1)
    assert heatmaps.dim(
    ) == ndims + 2, 'expected heatmaps to be a {}D tensor'.format(ndims + 2)
    assert heatmaps.size()[:-ndims] == mu_t.size()[:-1]

    gauss = make_gauss(mu_t, heatmaps.size()[2:], sigma_t)
    divergences = divergence(heatmaps, gauss, ndims)
    return divergences


def js_reg_losses(heatmaps, mu_t, sigma_t):
    return _divergence_reg_losses(heatmaps, mu_t, sigma_t, _js)


def average_loss(losses, mask=None):

    if mask is not None:
        assert mask.size() == losses.size(
        ), 'mask must be the same size as losses'
        losses = losses * mask
        denom = mask.sum()
    else:
        denom = losses.numel()

    # Prevent division by zero
    if isinstance(denom, int):
        denom = max(denom, 1)
    else:
        denom = denom.clamp(1)

    return losses.sum() / denom