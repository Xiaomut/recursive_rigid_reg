import torch
import torch.nn as nn

import sys

sys.path.append("../")
sys.path.append("./")
from metrics.function import normalized_cross_correlation


class NormalizedCrossCorrelation(nn.Module):
    """ N-dimensional normalized cross correlation (NCC)

    Args:
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def __init__(self, eps=1e-8, return_map=False, reduction='mean'):

        super(NormalizedCrossCorrelation, self).__init__()

        self._eps = eps
        self._return_map = return_map
        self._reduction = reduction

    def forward(self, x, y):

        return normalized_cross_correlation(x, y, self._return_map,
                                            self._reduction, self._eps)


def pearson_correlation(fixed, warped):
    flatten_fixed = torch.flatten(fixed, start_dim=1)
    flatten_warped = torch.flatten(warped, start_dim=1)

    mean1 = torch.mean(flatten_fixed)
    mean2 = torch.mean(flatten_warped)
    var1 = torch.mean((flatten_fixed - mean1)**2)
    var2 = torch.mean((flatten_warped - mean2)**2)

    cov12 = torch.mean((flatten_fixed - mean1) * (flatten_warped - mean2))
    eps = 1e-6
    pearson_r = cov12 / torch.sqrt((var1 + eps) * (var2 + eps))

    return pearson_r
