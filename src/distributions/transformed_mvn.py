import torch
import torch.distributions as td


class TransformedMultivariateNormal(td.TransformedDistribution):
    def __init__(self, mvn_base, transforms):
        super().__init__(mvn_base, transforms)

    @property
    def mean(self):
        """
        Returns:
        The transformation of the mean of the underlying MVN distribution. (BxCHW Tensor)
        Note that this is not strictly the mean of the transformed distribution
        """
        compose = td.transforms.ComposeTransform(self.transforms)
        return compose(self.base_dist.mean)

    @property
    def covariance_matrix(self):
        """
        Returns:
        The covariance matrix of the underlying MVN distribution (BxCHWxCHW Tensor)
        """
        return self.base_dist.covariance_matrix
