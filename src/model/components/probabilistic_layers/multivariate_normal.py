import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from distributions.lazy_low_rank_multivariate_normal_wrapper import (
    LazyLowRankMultivariateNormalWrapper,
)

# Add for numerical stability around zero
EPS = 1e-5


def _rank_scaled_eps(rank):
    power = 45 / rank
    power = min(5, power)
    return 10 ** (-power)


def _inverse_softplus(x):
    return torch.log(torch.exp(x) - 1.0)


class DiagonalMultivariateNormalLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        lazy=False,
        stabilize_mean=False,
        stabilize_cov=False,
        zero_diag=False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.zero_diag = zero_diag
        self.lazy = lazy

        self.mean_layer = nn.Linear(
            in_features=self.in_features, out_features=self.out_features
        )
        if not self.zero_diag:
            self.diag_layer = nn.Linear(
                in_features=self.in_features, out_features=self.out_features
            )

        if stabilize_mean:
            nn.init.normal_(self.mean_layer.weight, mean=0, std=EPS)
            nn.init.constant_(self.mean_layer.bias, 0.5)

        if stabilize_cov:
            if not self.zero_diag:
                nn.init.normal_(self.diag_layer.weight, mean=0, std=EPS)
                nn.init.constant_(
                    self.diag_layer.bias, _inverse_softplus(torch.ones(1) * EPS).item()
                )

    def forward(self, x, mean_only=False):
        if len(x.size()) != 2:
            raise RuntimeError(
                f"Incorrect dimensionality of input. Expected 2, got {len(x.size())}"
            )

        if x.size(1) != self.in_features:
            raise RuntimeError(
                f"Incorrect size of input in dim 2. Expected {self.in_features}, got {x.size(1)}"
            )

        mean = self.mean_layer(x)

        # Freeze/Unfreeze diag and factor weights
        if not self.zero_diag:
            self.diag_layer.weight.requires_grad = not mean_only
            self.diag_layer.bias.requires_grad = not mean_only

        # Calculate var
        if self.zero_diag:
            var = torch.zeros_like(mean) + EPS
        else:
            var = self.diag_layer(x)
            var = F.softplus(var) + EPS

        if self.lazy:
            return LazyLowRankMultivariateNormalWrapper(
                mean, torch.zeros_like(mean).unsqueeze(2), var
            )

        return td.LowRankMultivariateNormal(
            mean, torch.zeros_like(mean).unsqueeze(2), var
        )


class DiagonalMultivariateNormalLayer2d(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        lazy=False,
        stabilize_mean=False,
        stabilize_cov=False,
        zero_diag=False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.zero_diag = zero_diag
        self.lazy = lazy

        self.mean_conv = nn.Conv2d(self.in_features, self.out_features, kernel_size=1)
        if not self.zero_diag:
            self.diag_conv = nn.Conv2d(
                self.in_features, self.out_features, kernel_size=1
            )

        if stabilize_mean:
            nn.init.normal_(self.mean_conv.weight, mean=0, std=EPS)
            nn.init.constant_(self.mean_conv.bias, 0.5)

        if stabilize_cov:
            if not self.zero_diag:
                nn.init.normal_(self.diag_conv.weight, mean=0, std=EPS)
                nn.init.constant_(
                    self.diag_conv.bias, _inverse_softplus(torch.ones(1) * EPS).item()
                )

    def forward(self, logits, mean_only=False):
        if len(logits.size()) != 4:
            raise RuntimeError(
                f"Incorrect dimensionality of input. Expected 4, got {len(logits.size())}"
            )

        if logits.size(1) != self.in_features:
            raise RuntimeError(
                f"Incorrect size of input in dim 2. Expected {self.in_features}, got {logits.size(1)}"
            )

        # Flatten for use in distribution
        mean = self.mean_conv(logits).flatten(start_dim=1)  # [BxCHW]

        # Freeze/Unfreeze diag and factor weights
        if not self.zero_diag:
            self.diag_conv.weight.requires_grad = not mean_only
            self.diag_conv.bias.requires_grad = not mean_only

        # Calculate var
        if self.zero_diag:
            var = torch.zeros_like(mean) + EPS
        else:
            var = self.diag_conv(logits).flatten(start_dim=1)  # [BxCHW]
            var = F.softplus(var) + EPS

        if self.lazy:
            return LazyLowRankMultivariateNormalWrapper(
                mean, torch.zeros_like(mean).unsqueeze(2), var
            )

        return td.LowRankMultivariateNormal(
            mean, torch.zeros_like(mean).unsqueeze(2), var
        )


class LowRankMultivariateNormalLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        lazy=False,
        stabilize_mean=False,
        stabilise_cov=False,
        zero_diag=False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.zero_diag = zero_diag
        self.lazy = lazy

        self.mean_layer = nn.Linear(
            in_features=self.in_features, out_features=self.out_features
        )
        self.factor_layer = nn.Linear(
            in_features=self.in_features, out_features=self.rank * self.out_features
        )
        if not self.zero_diag:
            self.diag_layer = nn.Linear(
                in_features=self.in_features, out_features=self.out_features
            )

        if stabilize_mean:
            nn.init.normal_(self.mean_layer.weight, mean=0, std=EPS)
            nn.init.constant_(self.mean_layer.bias, 0.5)

        if stabilise_cov:
            nn.init.normal_(self.factor_layer.weight, mean=0, std=EPS)
            nn.init.constant_(self.factor_layer.bias, 0)

            if not self.zero_diag:
                nn.init.normal_(self.diag_layer.weight, mean=0, std=EPS)
                nn.init.constant_(
                    self.diag_layer.bias, _inverse_softplus(torch.ones(1) * EPS).item()
                )

    def forward(self, x, mean_only=False):
        if len(x.size()) != 2:
            raise RuntimeError(
                f"Incorrect dimensionality of input. Expected 2, got {len(x.size())}"
            )

        if x.size(1) != self.in_features:
            raise RuntimeError(
                f"Incorrect size of input in dim 2. Expected {self.in_features}, got {x.size(1)}"
            )

        # Freeze/Unfreeze diag and factor weights
        self.factor_layer.weight.requires_grad = not mean_only
        self.factor_layer.bias.requires_grad = not mean_only

        if not self.zero_diag:
            self.diag_layer.weight.requires_grad = not mean_only
            self.diag_layer.bias.requires_grad = not mean_only

        # Calculate mean
        mean = self.mean_layer(x)

        # Calculate cov factor and diag
        if self.zero_diag:
            cov_diag = torch.zeros_like(mean) + EPS
        else:
            cov_diag = self.diag_layer(x)
            cov_diag = F.softplus(cov_diag) + _rank_scaled_eps(self.rank)

        cov_factor = self.factor_layer(x).unflatten(-1, (self.rank, self.out_features))
        cov_factor = cov_factor.permute(0, 2, 1)

        if self.lazy:
            return LazyLowRankMultivariateNormalWrapper(mean, cov_factor, cov_diag)

        return td.LowRankMultivariateNormal(mean, cov_factor, cov_diag)


class LowRankMultivariateNormalLayer2d(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        lazy=False,
        stabilize_mean=False,
        stabilize_cov=False,
        zero_diag=False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.zero_diag = zero_diag
        self.lazy = lazy

        self.mean_conv = nn.Conv2d(
            self.in_features, 1 * self.out_features, kernel_size=1
        )
        self.factor_conv = nn.Conv2d(
            self.in_features, self.rank * self.out_features, kernel_size=1
        )
        if not self.zero_diag:
            self.diag_conv = nn.Conv2d(
                self.in_features, 1 * self.out_features, kernel_size=1
            )

        if stabilize_mean:
            nn.init.normal_(self.mean_conv.weight, mean=0, std=EPS)
            nn.init.constant_(self.mean_conv.bias, 0.5)

        if stabilize_cov:
            nn.init.normal_(self.factor_conv.weight, mean=0, std=EPS)
            nn.init.constant_(self.factor_conv.bias, 0)

            if not self.zero_diag:
                nn.init.normal_(self.diag_conv.weight, mean=0, std=EPS)
                nn.init.constant_(
                    self.diag_conv.bias, _inverse_softplus(torch.ones(1) * EPS).item()
                )

    def forward(self, logits, mean_only=False):
        if len(logits.size()) != 4:
            raise RuntimeError(
                f"Incorrect dimensionality of input. Expected 2, got {len(logits.size())}"
            )

        if logits.size(1) != self.in_features:
            raise RuntimeError(
                f"Incorrect size of input in dim 2. Expected {self.in_features}, got {logits.size(1)}"
            )

        # Freeze/Unfreeze diag and factor weights
        self.factor_conv.weight.requires_grad = not mean_only
        self.factor_conv.bias.requires_grad = not mean_only

        if not self.zero_diag:
            self.diag_conv.weight.requires_grad = not mean_only
            self.diag_conv.bias.requires_grad = not mean_only

        # Calculate mean
        mean = self.mean_conv(logits)
        mean = torch.flatten(mean, start_dim=1)  # [BxCHW]

        # Calculate cov factor and diag
        if self.zero_diag:
            cov_diag = torch.zeros_like(mean) + EPS
        else:
            cov_diag = self.diag_conv(logits)
            cov_diag = torch.flatten(cov_diag, start_dim=1)  # [BxCHW]
            cov_diag = F.softplus(cov_diag) + EPS  # _rank_scaled_eps(self.rank)

        cov_factor = self.factor_conv(logits)  # [BxRCxHxW]
        cov_factor = cov_factor.unflatten(
            1, (self.rank, cov_factor.size(1) // self.rank)
        )  # [BxRxCxHxW]
        cov_factor = torch.flatten(cov_factor, start_dim=2)  # [BxRxCHW]
        cov_factor = cov_factor.permute(0, 2, 1)

        if self.lazy:
            return LazyLowRankMultivariateNormalWrapper(mean, cov_factor, cov_diag)

        return td.LowRankMultivariateNormal(mean, cov_factor, cov_diag)
