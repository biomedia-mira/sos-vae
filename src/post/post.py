import warnings

import torch

from distributions.lazy_low_rank_multivariate_normal_wrapper import (
    LazyLowRankMultivariateNormalWrapper,
)


def _low_rank_svd(x):
    """
    Perform singular value decomposition on a tensor of shape BxCHWxR
    N.B: Returned in reverse principal order

    Parameters:
        x (BxCHWxR Tensor): tensor to be decomposed
    """

    xx = torch.einsum("...ij,...ik->...jk", x, x)
    s2, v = torch.symeig(xx, eigenvectors=True)
    s = torch.sqrt(s2)
    u = x @ (v / s.unsqueeze(-2))
    return u, s, v


def _avoid_zero(x, eps):
    sign = x.sign()
    # For this purpose we want zero to have a sign of +1
    sign[sign == 0] = 1
    return (x * sign).clamp(min=eps) * sign


@torch.no_grad()
def modify_dist(distribution, scales=None, diag=None, pca=False, stabilising_eps=0.03):
    """
    Reduce the noise (particularly of samples) of a given distribution.
        scales controls how much variation can be found in samples
        diag controls the noise of samples

    N.B: when scaling with PCA, zero-scaling cannot be stably achieved - as close as possible is used instead

    Parameters:
        distribution (LazyLowRankMultivariateNormalWrapper): distribution to modify
        scales (R Tensor): factors to scale the covariance factor by
        eps (float): value to replace all cov_diag values with
        pca (bool): whether scales should be applied to the covariance factor, or to its principle components

    Returns:
        Modified distribution (LazyLowRankMultivariateNormalWrapper)
    """
    if not (
        len(scales.shape) == 1 and scales.size(0) == distribution.cov_factor.size(-1)
    ):
        raise RuntimeError("Mismatched scales and distribution rank")

    if pca and scales is None:
        warnings.warn("PCA requested, but no scales given: no scaling applied")

    if scales is not None:
        if pca:
            u, s, v = _low_rank_svd(distribution.cov_factor)
            scales = _avoid_zero(scales, stabilising_eps).to(s.device)
            s = s * scales.flip(0)

            cov_factor = torch.matmul(
                torch.matmul(u, torch.diag_embed(s)), torch.transpose(v, 1, 2)
            )
            if torch.any(torch.isnan(cov_factor)):
                warnings.warn(
                    "NaN encountered in PCA. This should only happen when using a diagonal distribution"
                )
                cov_factor = torch.zeros_like(cov_factor)
        else:
            cov_factor = distribution.cov_factor * scales.view(1, 1, -1)
    else:
        cov_factor = distribution.cov_factor

    if diag is not None:
        cov_diag = torch.zeros_like(distribution.cov_diag) + diag
    else:
        cov_diag = distribution.cov_diag

    return LazyLowRankMultivariateNormalWrapper(distribution.mean, cov_factor, cov_diag)
