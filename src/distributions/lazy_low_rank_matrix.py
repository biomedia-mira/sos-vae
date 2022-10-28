import torch
from torch.distributions.utils import lazy_property


class LazyLowRankMatrix:
    def __init__(self, cov_factor, cov_diag):
        """
        Parameters:
            cov_factor (BxCHWxR Tensor)
            cov_diag(BxCHW Tensor)
        """
        assert len(cov_factor.shape) - 1 == len(
            cov_diag.shape
        ), "Mismatched cov_factor and cov_diag dimensions"
        if len(cov_diag.shape) == 1:
            cov_factor = cov_factor.unsqueeze(0)
            cov_diag = cov_diag.unsqueeze(0)

        assert (
            len(cov_diag.shape) == 2
        ), "Incorrect dimensionality of cov_factor and cov_diag"

        assert cov_factor.size(2) <= cov_factor.size(
            1
        ), "Rank is greater than dimension"

        self.cov_factor = cov_factor
        self.cov_diag = cov_diag

    @lazy_property
    def full(self):
        """
        Returns:
            Full matrix (BxCHWxCHW Tensor)

        N.B: In many cases this may not fit in available memory
        """
        return torch.matmul(
            self.cov_factor, torch.transpose(self.cov_factor, -2, -1)
        ) + torch.diag_embed(self.cov_diag)

    def row(self, index):
        """
        Calculate a single row of the matrix

        Parameters:
            index (int): the index of the row to calculate

        Returns:
            Row of the matrix (BxCHW Tensor)
        """
        row = torch.matmul(
            self.cov_factor[:, index, :].unsqueeze(1),
            torch.transpose(self.cov_factor, 1, 2),
        ).squeeze(1)
        row[:, index] += self.cov_diag[:, index]
        return row

    @lazy_property
    def diag(self):
        """
        Calculate the diagonal elements of the matrix. This is the variance if the matrix is a covariance matrix

        Returns:
            Diagonal entries (BxCHW Tensor)
        """
        return self.cov_factor.pow(2).sum(-1) + self.cov_diag
