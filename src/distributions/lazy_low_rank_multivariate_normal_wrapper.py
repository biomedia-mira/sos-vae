import torch
import torch.distributions as td
from torch.distributions.utils import lazy_property, _standard_normal

from distributions.lazy_low_rank_matrix import LazyLowRankMatrix


class LazyLowRankMultivariateNormalWrapper:
    def __init__(self, mean, cov_factor, cov_diag):
        """
        Parameters:
            mean (BxCHW Tensor)
            cov_factor (BxCHWxR Tensor)
            cov_diag (BxCHW Tensor)
        """
        # After these transformations and assertions, all variables are batched

        assert len(mean.shape) <= 2, "Invalid number of dimensions for mean"
        assert torch.tensor(mean.shape).equal(
            torch.tensor(cov_diag.shape)
        ), "Mismatched mean and covariance dimensions"

        if len(mean.shape) == 1:
            mean = mean.unsqueeze(0)

        self.mean = mean
        self.low_rank_cov = LazyLowRankMatrix(cov_factor, cov_diag)

    def __len__(self):
        return self.mean.size(0)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise NotImplementedError("Non integer indeces are not supported")

        # Batch dims will be added in constructor call
        return LazyLowRankMultivariateNormalWrapper(
            self.mean[index], self.cov_factor[index], self.cov_diag[index]
        )

    def unflattened_mean(self, shape):
        """
        Unflatten the mean into specified shape

        Parameters:
            shape (tuple of int). N.B: does not include batch dimension!

        Returns:
            Unflattened mean (BxShape Tensor)
        """
        return self.mean.unflatten(1, shape)

    def cpu(self):
        """
        Return copy of with all tensors on the cpu
        """
        return LazyLowRankMultivariateNormalWrapper(
            self.mean.cpu(), self.cov_factor.cpu(), self.cov_diag.cpu()
        )

    def cuda(self):
        """
        Return copy of with all tensors on the cuda
        """
        return LazyLowRankMultivariateNormalWrapper(
            self.mean.cuda(), self.cov_factor.cuda(), self.cov_diag.cuda()
        )

    @property
    def cov_diag(self):
        return self.low_rank_cov.cov_diag

    @property
    def cov_factor(self):
        return self.low_rank_cov.cov_factor

    @property
    def rank(self):
        return self.cov_factor.size(-1)

    @lazy_property
    def covariance_matrix(self):
        """
        Returns:
            Full covariance matrix (BxCHWxCHW Tensor)

        N.B: In many cases this may not fit in available memory
        """
        return self.low_rank_cov.full

    @lazy_property
    def variance(self):
        """
        Calculate the variance

        Returns:
            Variance (BxCHW Tensor)
        """
        return self.low_rank_cov.diag

    @lazy_property
    def pixel_influence_factor(self):
        """
        Calculate the pixel influence factor (defined for a pixel
        as the sum of the squares of its corresponding covariances)

        Returns:
            Pixel influence factor (BxCHW Tensor)
        """
        pixel_influence_factor = torch.zeros_like(self.mean)
        for pixel in range(self.mean.size(1)):
            cov_row = self.row(pixel)
            squared_cov_row = torch.square(cov_row)
            pixel_influence_factor[:, pixel] = torch.sum(squared_cov_row, dim=-1)

        return pixel_influence_factor

    def row(self, index):
        """
        Calculate a single row of the covariance matrix

        Parameters:
            index (int): the index of the row to calculate

        Returns:
            Row of the covariance matrix (BxCHW Tensor)
        """
        return self.low_rank_cov.row(index)

    @lazy_property
    def td_distribution(self):
        """
        Create a torch LowRankMultivariateNormal from this representation

        Returns:
            torch.distributions.LowRankMultivariateNormal
        """
        return td.LowRankMultivariateNormal(self.mean, self.cov_factor, self.cov_diag)

    @property
    def batch_size(self):
        return self.mean.size(0)

    def sample(self, omega_p=None, omega_d=None):
        """
        Sample the distribution.
        N.B: this always returns one sample for each batch item.

        Parameters:
            omega_p (BxR Tensor): auxiliary variable for the covariance factor
            om_f (BxCHW Tensor): auxiliary variable for the covariance diagonal
        """
        if omega_p is not None:
            if (
                len(omega_p.shape) != 2
                or omega_p.size(0) != self.batch_size
                or omega_p.size(1) != self.rank
            ):
                raise RuntimeError(
                    f"Incorrect dimensionality for omega_p. Got {omega_p.shape}, expected ({self.batch_size}, {self.rank})"
                )
            omega_p = omega_p.to(self.mean.device)
        else:
            omega_p = _standard_normal(
                torch.Size((self.batch_size, self.rank)),
                self.mean.dtype,
                self.mean.device,
            )

        if omega_d is not None:
            if (
                len(omega_d.shape) != 2
                or omega_d.size(0) != self.batch_size
                or omega_d.size(1) != self.mean.size(-1)
            ):
                raise RuntimeError(
                    f"Incorrect dimensionality for omega_d. Got {omega_d.shape}, expected ({self.batch_size}, {self.mean.size(-1)})"
                )
            omega_d = omega_d.to(self.mean.device)
        else:
            omega_d = _standard_normal(
                torch.Size((self.batch_size, self.mean.size(-1))),
                self.mean.dtype,
                self.mean.device,
            )

        return (
            self.mean
            + torch.matmul(self.cov_factor, omega_p.unsqueeze(-1)).squeeze(-1)
            + self.cov_diag.sqrt() * omega_d
        )
