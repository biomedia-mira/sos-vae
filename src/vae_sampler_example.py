import torch
import torch.distributions as td

import matplotlib
import matplotlib.pyplot as plt

from model.vae import VAE

from visualiser.visualiser import (
    visualise_distribution,
    visualise_sample_variation,
    visualise_observation_space,
    visualise_scaled_pca_components,
)

from interactive.vae_sampler import launch as launch_vae_sampler

torch.autograd.set_grad_enabled(False)

if __name__ == "__main__":
    matplotlib.use("TkAgg")

    NUM_SHOW = 20
    GRAYSCALE = False
    SHAPE = (1 if GRAYSCALE else 3, 218, 178)
    RANK = 25
    LATENT_FEATURES = 128

    DEVICE = "cuda"

    model = VAE(LATENT_FEATURES, RANK, 1 if GRAYSCALE else 3)
    model.load_state_dict(torch.load("out/model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    latent_prior = td.MultivariateNormal(
        torch.zeros(LATENT_FEATURES, device=DEVICE),
        torch.eye(LATENT_FEATURES, device=DEVICE),
    )

    latent_samples = latent_prior.sample((NUM_SHOW,))

    distribution = model.decode(latent_samples)

    visualise_distribution(
        distribution,
        None,
        NUM_SHOW,
        img_size=SHAPE,
        title="Latent Samples",
    )
    plt.show()

    visualise_sample_variation(distribution, NUM_SHOW, NUM_SHOW // 2, SHAPE)
    plt.show()

    visualise_observation_space(distribution, NUM_SHOW // 2, SHAPE)
    plt.show()

    visualise_scaled_pca_components(distribution, SHAPE)
    plt.show()

    launch_vae_sampler(model, latent_prior, RANK, SHAPE)
