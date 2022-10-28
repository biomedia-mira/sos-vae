import torch
import torch.nn as nn

from model.vae import VAE


class ConditionalVAE(VAE):
    """
    Implementation of a Conditional Stochastic VAE with a ResNet architecture
    """

    def __init__(self, latent_features, decoder_rank, channels, num_labels):
        super().__init__(latent_features, decoder_rank, channels, latent_channels=2)

        self.num_labels = num_labels
        self.label_embedding = nn.Embedding(self.num_labels, self.latent_features)

    def decode(self, z, label, mean_only=False):
        """
        Run just the decoder on a batch of latent samples

        Parameters:
            z (BxF Tensor): Batch of latent samples
            label (B Tensor): Batch of labels
            mean_only (boolean): True for pre training

        Returns:
            distribution (LazyLowRankMultivariateNormalWrapper)
        """
        label_embed = self.label_embedding(label)
        combined = torch.cat((z, label_embed), dim=1)

        return super().decode(combined, mean_only=mean_only)

    def forward(self, x, mean_only=False):
        # N.B: Only 1 MC sample needed for sufficiently large batch size

        x, label = x

        enc_distribution = self.encoder(x)

        # Sample from probabilistic encoder with reparameterisation trick
        z = enc_distribution.rsample()

        distribution = self.decode(z, label, mean_only=mean_only)

        return enc_distribution, distribution
