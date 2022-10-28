import warnings
import numpy as np

import torch

from torchvision.utils import make_grid, save_image

import matplotlib
import matplotlib.pyplot as plt

from post.post import modify_dist

matplotlib.use("Agg")

from corrections.corrections import propagated_correction
from corrections.mask_generation import random_patch

from model.vae import VAE
from model.conditional_vae import ConditionalVAE

from post.lerp import slerp


@torch.no_grad()
def visualise_distribution(
    distribution,
    original_data,
    num_show,
    img_size=None,
    patch_size=None,
    title=None,
    transform=lambda x: x,
):
    """
    Create visualised distribution figure

    Parameters:
        distribution (LazyLowRankMultivariateNormalWrapper): distribution to visualise
        original_data (BxCxHxW Tensor): original data to compare against
        num_show (int): number of data points to show in figure (must be less than or equal to batch size)
        patch_size (int or (int, int)): patch size for correction
        img_size (tuple of int): size of images (inferred from original_data if provided) (B,H,W)
        title (string): title of figure
        transform(function): transformation to be applied to convert network output into final RGB images (e.g. colour space conversion)

    Returns:
        matplotlib.figure.Figure
    """

    td_distribution = distribution.td_distribution
    batch_size = distribution.batch_size

    if num_show > batch_size:
        raise RuntimeError(
            "Asking to display more images than available in a single batch"
        )

    if original_data is None:
        if img_size is None:
            raise RuntimeError("No img_size argument provided")

        total_plots = 6
        plt.figure(figsize=(num_show, total_plots))
    else:
        original_data = original_data.detach().cpu()

        img_size = tuple(original_data.shape)[1:]
        if patch_size is None:
            raise RuntimeError("No patch_size argument provided")

        total_plots = 11
        plt.figure(figsize=(num_show, total_plots))

    # Calculations
    means = distribution.mean.detach().cpu()
    samples = td_distribution.sample().detach().cpu()
    sample_diff = samples - means

    if original_data is not None:
        # Calculate residuals
        residuals = original_data[:num_show] - means.unflatten(1, img_size)[:num_show]
        residuals = residuals.abs()

        # Calculate patch and correction
        mask = random_patch(distribution, original_data, patch_size).detach().cpu()
        patch = original_data * mask
        corrected_means = (
            propagated_correction(distribution, original_data, mask).detach().cpu()
        )

        diff = corrected_means - means

    plt.clf()
    if title is not None:
        plt.suptitle(title)
    plt.subplots_adjust(
        left=0.01, right=0.98, top=0.933, bottom=0.01, hspace=0.05, wspace=0
    )

    plt_counter = 0
    if original_data is not None:
        # Original data
        plt_counter += 1
        plt.subplot(total_plots, 1, plt_counter)
        od_plt = original_data[:num_show]
        od_plt = transform(od_plt)
        od_plt = make_grid(od_plt, nrow=num_show)
        od_plt = od_plt.cpu().numpy()
        od_plt = np.transpose(od_plt, (1, 2, 0))

        plt.ylabel("Original")
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.imshow(od_plt)

    # Means
    plt_counter += 1
    plt.subplot(total_plots, 1, plt_counter)
    means_plt = means.unflatten(1, img_size)[:num_show]
    means_plt = transform(means_plt)
    means_plt = means_plt.clamp(0, 1)
    means_plt = make_grid(means_plt, nrow=num_show)
    means_plt = means_plt.cpu().numpy()
    means_plt = np.transpose(means_plt, (1, 2, 0))

    plt.ylabel("Mean")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(means_plt)

    if original_data is not None:
        # Residual
        plt_counter += 1
        plt.subplot(total_plots, 1, plt_counter)
        res_plt = make_grid(residuals, nrow=num_show)
        res_plt = res_plt.cpu().numpy()
        res_plt = np.transpose(res_plt, (1, 2, 0))
        # To preserve scaling and show pixelwise, average and remove colour dim
        res_plt = res_plt.mean(axis=2)

        plt.ylabel("Residual")
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.imshow(res_plt)

    # Samples
    plt_counter += 1
    plt.subplot(total_plots, 1, plt_counter)
    sam_plt = samples.unflatten(1, img_size)[:num_show]
    sam_plt = transform(sam_plt)
    sam_plt = sam_plt.clamp(0, 1)
    sam_plt = make_grid(sam_plt, nrow=num_show)
    sam_plt = sam_plt.cpu().numpy()
    sam_plt = np.transpose(sam_plt, (1, 2, 0))

    plt.ylabel("Sample")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(sam_plt)

    # Sample Diff
    plt_counter += 1
    plt.subplot(total_plots, 1, plt_counter)
    sam_dif_plt = sample_diff.unflatten(1, img_size)[:num_show]
    sam_dif_plt = make_grid(sam_dif_plt, nrow=num_show)
    sam_dif_plt = sam_dif_plt.cpu().numpy()
    sam_dif_plt = np.transpose(sam_dif_plt, (1, 2, 0))
    # To preserve scaling and show pixelwise, average and remove colour dim
    sam_dif_plt = sam_dif_plt.mean(axis=2)

    plt.ylabel("Sample Diff")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(sam_dif_plt)

    # Variance
    plt_counter += 1
    plt.subplot(total_plots, 1, plt_counter)
    # Extract diagonal: variance
    var = distribution.variance
    var_plt = var.unflatten(1, img_size)[:num_show]
    var_plt = make_grid(var_plt, nrow=num_show)
    var_plt = var_plt.cpu().numpy()
    var_plt = np.transpose(var_plt, (1, 2, 0))
    # To preserve scaling and show pixelwise, average and remove colour dim
    var_plt = var_plt.mean(axis=2)

    plt.ylabel("Variance")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(var_plt)

    # Covariance of middle pixel
    # Mid pixel
    mid = img_size[-2] // 2 * img_size[-1] + img_size[-1] // 2
    # Extract row
    cov = distribution.row(mid)
    # Remove middle pixel as it is the variance and may be very different, ruining the scale
    cov[:, mid] = 0
    cov_plt = cov.unflatten(1, img_size)[:num_show]

    pos_cov_plt = cov_plt.clamp(0, cov_plt.max())
    neg_cov_plt = cov_plt.clamp(cov_plt.min(), 0) * -1

    # Plot for pos
    plt_counter += 1
    plt.subplot(total_plots, 1, plt_counter)

    pos_cov_plt = make_grid(pos_cov_plt, nrow=num_show)
    pos_cov_plt = pos_cov_plt.cpu().numpy()
    pos_cov_plt = np.transpose(pos_cov_plt, (1, 2, 0))
    # To preserve scaling and show pixelwise, average and remove colour dim
    pos_cov_plt = pos_cov_plt.mean(axis=2)

    plt.ylabel("+Mid Cov")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(pos_cov_plt)

    # Plot for neg
    plt_counter += 1
    plt.subplot(total_plots, 1, plt_counter)

    neg_cov_plt = make_grid(neg_cov_plt, nrow=num_show)
    # Convert greyscale to one channel
    neg_cov_plt = neg_cov_plt.cpu().numpy()
    neg_cov_plt = np.transpose(neg_cov_plt, (1, 2, 0))
    # To preserve scaling and show pixelwise, average and remove colour dim
    neg_cov_plt = neg_cov_plt.mean(axis=2)

    plt.ylabel("-Mid Cov")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(neg_cov_plt)

    if original_data is not None:
        # Patch
        plt_counter += 1
        plt.subplot(total_plots, 1, plt_counter)
        patch_plt = patch[:num_show]
        patch_plt = transform(patch_plt) * mask[:num_show]
        patch_plt = make_grid(patch_plt, nrow=num_show)
        patch_plt = patch_plt.cpu().numpy()
        patch_plt = np.transpose(patch_plt, (1, 2, 0))

        plt.ylabel("Patch")
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.imshow(patch_plt)

        # Corrected Means
        plt_counter += 1
        plt.subplot(total_plots, 1, plt_counter)
        cor_means_plt = corrected_means.unflatten(1, img_size)[:num_show]
        cor_means_plt = transform(cor_means_plt)
        cor_means_plt = cor_means_plt.clamp(0, 1)
        cor_means_plt = make_grid(cor_means_plt, nrow=num_show)
        cor_means_plt = cor_means_plt.cpu().numpy()
        cor_means_plt = np.transpose(cor_means_plt, (1, 2, 0))

        plt.ylabel("Corrected")
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.imshow(cor_means_plt)

        # Diff
        plt_counter += 1
        plt.subplot(total_plots, 1, plt_counter)
        diff_plt = diff.unflatten(1, img_size)[:num_show]
        diff_plt = make_grid(diff_plt, nrow=num_show)
        diff_plt = diff_plt.cpu().numpy()
        diff_plt = np.transpose(diff_plt, (1, 2, 0))
        # To preserve scaling and show pixelwise, average and remove colour dim
        diff_plt = diff_plt.mean(axis=2)

        plt.ylabel("Diff")
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.imshow(diff_plt)

    return plt.gcf()


def generate_new_results(
    model,
    latent_prior,
    num_show,
    img_size,
    labels=None,
    title=None,
    mean_only=False,
    transform=lambda x: x,
):
    """
    Create visualised figure of distribution from latent samples

    Parameters:
        model: generative model to decode latent samples
        latent_prior (td.distribution): latent prior from which to sample latent samples
        num_show (int): number of data points to show in figure (must be less than or equal to batch size)
        img_size (int or tuple of int): size of images (B,H,W)
        labels (Tensor): tensor of length num_show containing labels, given only for conditional models
        title (string): title of figure
        mean_only (boolean): True for pre-training
        transform(function): transformation to be applied to convert network output into final RGB images (e.g. colour space conversion)

    Returns:
        matplotlib.figure.Figure
    """

    if labels is not None and not isinstance(model, ConditionalVAE):
        warnings.warn(
            "labels and non-conditional model provided. labels will be ignored"
        )
    if isinstance(model, ConditionalVAE) and labels is None:
        raise RuntimeError("Conditional model and no labels provided")

    if isinstance(model, ConditionalVAE):
        if len(labels.shape) != 1 or labels.size(0) != num_show:
            raise RuntimeError(
                f"Invalid number of labels given. Expected [{num_show}], received {labels.shape}"
            )

    latent_samples = latent_prior.sample((num_show,))

    with torch.no_grad():
        if isinstance(model, ConditionalVAE):
            distribution = model.decode(latent_samples, labels, mean_only=mean_only)
        else:
            distribution = model.decode(latent_samples, mean_only=mean_only)

    return visualise_distribution(
        distribution,
        None,
        num_show,
        img_size=img_size,
        title=title,
        transform=transform,
    )


def visualise_observation_space(
    distribution, num_steps, img_size, sample_scaling=1.0, transform=lambda x: x
):
    """
    Visualise a slice of the observation space by taking four random samples from the distribution
    and interpolating across them in two dimensions.
    Interpolations are carried out on the auxiliary noise variables

    Parameters:
        distribution (LazyLowRankMultivariateNormalWrapper): distribution to visualise
        num_steps (int): the number of interpolation steps between the corners in both dimensions
        img_size (int or tuple of int): size of images (B,H,W)
        sample_scaling (float): Coefficient to scale the auxiliary noise variables by, if desired
        transform(function): transformation to be applied to convert network output into final RGB images (e.g. colour space conversion)

    Returns:
        matplotlib.figure.Figure
    """
    distribution = distribution[0]

    tl = torch.randn(1, distribution.rank) * sample_scaling
    tr = torch.randn(1, distribution.rank) * sample_scaling
    bl = torch.randn(1, distribution.rank) * sample_scaling
    br = torch.randn(1, distribution.rank) * sample_scaling

    grid = torch.empty(num_steps + 2, num_steps + 2, torch.tensor(img_size).prod())

    for i in range(num_steps + 2):
        for j in range(num_steps + 2):
            xt = slerp(tl, tr, i / (num_steps + 1))
            xb = slerp(bl, br, i / (num_steps + 1))
            loc = slerp(xt, xb, j / (num_steps + 1))

            grid[i, j] = distribution.sample(omega_p=loc)

    grid = grid.flatten(start_dim=0, end_dim=1)
    grid = grid.unflatten(1, img_size)
    grid = transform(grid)

    plt.figure()

    os_plt = grid.clamp(0, 1)
    os_plt = make_grid(os_plt, nrow=num_steps + 2)
    os_plt = os_plt.cpu().numpy()
    os_plt = np.transpose(os_plt, (1, 2, 0))

    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(os_plt)

    return plt.gcf()


@torch.no_grad()
def visualise_scaled_pca_components(
    distribution,
    img_size,
    min_scale=-5,
    max_scale=5,
    scale_interval=0.5,
    omega_p=None,
    transform=lambda x: x,
):
    """
    Visualise the effect of scaling each component of the PCA decomposed covariance factor.

    Parameters:
        distribution (LazyLowRankMultivariateNormalWrapper): distribution to visualise
        img_size (int or tuple of int): size of images (B,H,W)
        min_scale (float): minimum factor to scale component by
        max_scale (float): maximum factor to scale component by
        scale_interval (float): interval between scale factors
        transform(function): transformation to be applied to convert network output into final RGB images (e.g. colour space conversion)

    Returns:
        matplotlib.figure.Figure
    """
    distribution = distribution[0].cuda()

    num_steps = int((max_scale - min_scale) // scale_interval + 1)
    grid = torch.empty(
        distribution.rank,
        num_steps,
        torch.tensor(img_size).prod(),
        device=distribution.mean.device,
    )

    if omega_p is None:
        omega_p = torch.randn((1, distribution.rank), device=distribution.mean.device)

    for r in range(distribution.rank):
        for i in range(num_steps):
            t = min_scale + i * scale_interval

            scales = torch.zeros(distribution.rank, device=distribution.mean.device)
            scales[r] = t

            modded_dist = modify_dist(distribution, scales, pca=True)
            grid[r, i] = modded_dist.sample(omega_p=omega_p)

    grid = grid.flatten(start_dim=0, end_dim=1)
    grid = grid.unflatten(1, img_size)
    grid = transform(grid)

    plt.figure()

    scaled_plt = grid.clamp(0, 1)
    scaled_plt = make_grid(scaled_plt, nrow=num_steps)
    scaled_plt = scaled_plt.cpu().numpy()
    scaled_plt = np.transpose(scaled_plt, (1, 2, 0))

    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(scaled_plt)

    return plt.gcf()


@torch.no_grad()
def visualise_sample_variation(
    distribution,
    num_show,
    num_samples,
    img_size,
    cols_per_batch_item=1,
    transform=lambda x: x,
):
    """
    Create figure showing many samples from a distribution

    Parameters:
        distribution (LazyLowRankMultivariateNormalWrapper): distribution to visualise
        num_show (int): number of data points to show in figure (must be less than or equal to batch size)
        num_samples (int): number of data points to show in figure (must be less than or equal to batch size)
        img_size (tuple of int): size of images (B,H,W)
        transform(function): transformation to be applied to convert network output into final RGB images (e.g. colour space conversion)

    Returns:
        matplotlib.figure.Figure
    """

    td_distribution = distribution.td_distribution
    batch_size = distribution.batch_size

    if num_show > batch_size:
        raise RuntimeError(
            "Asking to display more images than available in a single batch"
        )

    means = distribution.mean.detach().cpu()[:num_show]
    means = means.unsqueeze(0).repeat(cols_per_batch_item, 1, 1).permute(1, 0, 2)

    samples = (
        td_distribution.sample((num_samples * cols_per_batch_item,))
        .detach()
        .cpu()[:, :num_show]
    )
    samples = samples.permute(1, 0, 2)

    means = means.flatten(start_dim=0, end_dim=1)
    samples = (
        samples.unflatten(1, (num_samples, cols_per_batch_item))
        .transpose(0, 1)
        .flatten(start_dim=0, end_dim=-2)
    )

    plt.figure()

    combi_plt = torch.cat([means, samples], dim=0)
    combi_plt = combi_plt.unflatten(1, img_size)
    combi_plt = transform(combi_plt)
    combi_plt = combi_plt.clamp(0, 1)
    combi_plt = make_grid(combi_plt, nrow=num_show * cols_per_batch_item)
    combi_plt = combi_plt.cpu().numpy()
    combi_plt = np.transpose(combi_plt, (1, 2, 0))

    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(combi_plt)

    return plt.gcf()
