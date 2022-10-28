import torch
import torch.nn as nn

from model.components.probabilistic_layers.multivariate_normal import (
    DiagonalMultivariateNormalLayer,
    LowRankMultivariateNormalLayer2d,
)

from model.components.residual import ResidualBlock, ResidualBlockTranspose

DATA_DIMS = (218, 178)

# Calculate size after convs
CONV_OUT_SIZE = torch.tensor(DATA_DIMS)
CONV_OUT_SIZE = torch.ceil(CONV_OUT_SIZE / 2.0).int()
CONV_OUT_SIZE = torch.ceil(CONV_OUT_SIZE / 2.0).int()
CONV_OUT_SIZE = torch.ceil(CONV_OUT_SIZE / 2.0).int()


class VAE(nn.Module):
    """
    Implementation of a Structured Observation Space VAE with a ResNet architecture
    """

    def __init__(self, latent_features, decoder_rank, channels, latent_channels=1):
        super().__init__()

        self.latent_features = latent_features
        self.decoder_rank = decoder_rank
        self.latent_channels = latent_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # <- CONV_OUT_SIZE
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            DiagonalMultivariateNormalLayer(512, self.latent_features),
        )

        self.decoder_base = nn.Sequential(
            nn.Unflatten(1, (self.latent_channels * self.latent_features, 1, 1)),
            nn.ConvTranspose2d(
                self.latent_channels * self.latent_features,
                self.latent_channels * self.latent_features,
                kernel_size=(CONV_OUT_SIZE[0], CONV_OUT_SIZE[1]),
                stride=1,
            ),
            # <- CONV_OUT_SIZE
            nn.Conv2d(self.latent_channels * self.latent_features, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResidualBlock(512, 512, stride=1),
            ResidualBlockTranspose(512, 256, stride=2, output_padding=0),
            ResidualBlock(256, 256, stride=1),
            ResidualBlockTranspose(256, 128, stride=2, output_padding=0),
            ResidualBlock(128, 128, stride=1),
            ResidualBlockTranspose(128, 64, stride=2, output_padding=1),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.decoder_out = LowRankMultivariateNormalLayer2d(
            64,
            channels,
            rank=self.decoder_rank,
            lazy=True,
            stabilize_mean=True,
            stabilize_cov=True,
            zero_diag=True,
        )

    def decode(self, z, mean_only=False):
        """
        Run just the decoder on a batch of latent samples

        Parameters:
            z (BxF Tensor): Batch of latent samples
            mean_only (boolean): True for pre training

        Returns:
            distribution (LazyLowRankMultivariateNormalWrapper)
        """
        distribution = self.decoder_base(z)
        distribution = self.decoder_out(distribution, mean_only=mean_only)
        return distribution

    def forward(self, x, mean_only=False):
        # N.B: Only 1 MC sample needed for sufficiently large batch size

        enc_distribution = self.encoder(x)

        # Sample from probabilistic encoder with reparameterisation trick
        z = enc_distribution.rsample()

        distribution = self.decode(z, mean_only=mean_only)

        return enc_distribution, distribution
