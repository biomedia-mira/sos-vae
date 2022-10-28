import torch


def propagated_correction(distribution, ground_truth, mask):
    """
    Replace the mean of distribution given ground truth where mask specifies (propagating using covariance).

    Parameters:
        distribution (LazyLowRankMultivariateNormalWrapper): distribution with mean to change.
        ground_truth (BxCxHxW Tensor): ground truth to correct to
        mask (BxCxHxW Tensor): mask indicating where in the ground_truth should be copied

    Returns:
        The updated means (BxCHW Tensor)
    """

    device = distribution.mean.device

    ground_truth = ground_truth.to(distribution.mean.device.type)
    mask = mask.to(distribution.mean.device.type)

    mean = distribution.mean  # shape [BxCHW]
    cov_factor = distribution.cov_factor  # shape [BxCHWxR]
    cov_diag = distribution.cov_diag  # shape [BxCHW]
    batch_size = mean.size(0)

    correct = ground_truth[mask]  # N.B: flattened
    correct = correct.unflatten(0, (batch_size, correct.size(0) // batch_size))

    mask = mask.flatten(start_dim=1)

    N = mean.size(1)  # total number of pixels = CHW
    q = N - correct.size(1)  # number of uncorrected pixels

    mu_1 = mean[~mask].unflatten(0, (batch_size, q))
    mu_2 = mean[mask].unflatten(0, (batch_size, N - q))

    cov_factor_1 = cov_factor[~mask, :].unflatten(0, (batch_size, q))
    cov_factor_2 = cov_factor[mask, :].unflatten(0, (batch_size, N - q))

    # Don't need to calculate cov_diag_1
    cov_diag_2 = cov_diag[mask].unflatten(0, (batch_size, N - q))
    cov_diag_2 = torch.diag_embed(cov_diag_2)

    sigma_12 = torch.matmul(cov_factor_1, torch.transpose(cov_factor_2, -1, -2))
    sigma_22 = (
        torch.matmul(cov_factor_2, torch.transpose(cov_factor_2, -1, -2)) + cov_diag_2
    )

    # x_1 | x_2 = correct ~ N(mu_bar,sigma_bar)
    mu_bar = mu_1 + torch.matmul(
        torch.matmul(sigma_12, torch.inverse(sigma_22)),
        (correct - mu_2).unsqueeze(2),
    ).squeeze(2)

    updated_means = torch.empty(batch_size, N, device=device)
    updated_means[mask] = correct.flatten()
    updated_means[~mask] = mu_bar.flatten()

    return updated_means
