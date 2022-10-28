import torch
import torch.nn.functional as F

# N.B: For training, all masks must generate with a fixed number of True values

# N.B: This file only supports 2D images


def _checks(distribution, ground_truth):
    """
    Performs argument validation checks. Raises RuntimeError on fail

    Parameters:
        distribution (LazyLowRankMultivariateNormalWrapper): distribution with mean to change.
        ground_truth (BxCxHxW Tensor): ground truth to correct to
    """
    if (
        torch.tensor(distribution.mean.shape).prod()
        != torch.tensor(ground_truth.shape).prod()
    ):
        raise RuntimeError("Mismatched sizes between prediction and ground truth")


def top_left_patch(distribution, ground_truth, patch_size):
    """
    Creates a mask in the shape of ground_truth with a patch of patch_size masked in the top left corner

    Parameters:
        distribution (LazyLowRankMultivariateNormalWrapper): distribution with mean to change.
        ground_truth (BxCxHxW Tensor): ground truth to correct to
        patch_size (int or (int, int)):

    Returns:
        A mask in the shape of ground_truth with a patch of patch_size masked in the top left corner
    """
    _checks(distribution, ground_truth)

    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size)

    patch_locations = (0, 0)

    mask = torch.zeros_like(ground_truth, dtype=torch.bool)
    mask[
        :, :, patch_locations[0] : patch_size[0], patch_locations[1] : patch_size[1]
    ] = True

    return mask


def random_patch(distribution, ground_truth, patch_size):
    """
    Creates a mask in the shape of ground_truth with a patch of patch_size masked in a random location

    Parameters:
        distribution (LazyLowRankMultivariateNormalWrapper): distribution with mean to change.
        ground_truth (BxCxHxW Tensor): ground truth to correct to
        patch_size (int or (int, int)):

    Returns:
        A mask in the shape of ground_truth with a patch of patch_size masked in a random location
    """
    _checks(distribution, ground_truth)

    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size)

    batch_size = distribution.mean.size(0)

    patch_locations = (
        torch.randint((ground_truth.size(-2) - patch_size[0]), (batch_size, 1)),
        torch.randint((ground_truth.size(-1) - patch_size[1]), (batch_size, 1)),
    )
    patch_locations = torch.cat(patch_locations, dim=1)  # [Bx2]

    mask = torch.zeros_like(ground_truth, dtype=torch.bool)

    for i in range(batch_size):
        mask[
            i,
            :,
            patch_locations[i, 0] : patch_locations[i, 0] + patch_size[0],
            patch_locations[i, 1] : patch_locations[i, 1] + patch_size[1],
        ] = True

    return mask


@torch.no_grad()
def unravel_index(index, shape):
    """
    Get the location of an index from a flattened tensor relative to an unflattened tensor

    Parameters:
        index (int): original index (in flattened tensor)
        shape (tuple of int): shape of unflattened tensor

    Returns:
        The location of the index in the unflattened tensor (tuple of int)
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def uncertain_patch(distribution, ground_truth, patch_size, metric="variance"):
    _checks(distribution, ground_truth)

    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size)

    if metric == "variance":
        uncertainty = distribution.variance  # [BxCHW]
    elif metric == "pixel influence":
        uncertainty = distribution.pixel_influence_factor
    else:
        raise ValueError("Metric not supported. Use 'variance' or 'pixel influece'")

    uncertainty = uncertainty.view_as(ground_truth)  # [BxCxHxW]
    batch_size = uncertainty.size(0)

    # need to check these three lines
    kernel = torch.ones(
        uncertainty.size(1), uncertainty.size(1), patch_size[0], patch_size[1]
    )
    patch_uncertainty = F.conv2d(uncertainty, kernel)  # BxCx(H-A+1)x(W-B+1)
    patch_uncertainty = torch.flatten(
        patch_uncertainty, start_dim=2
    )  # BxCx(H-A+1)(W-B+1)
    patch_uncertainty = patch_uncertainty.mean(dim=1)  # Bx(H-A+1)(W-B+1)
    patch_locations_flat = torch.argmax(patch_uncertainty, dim=1, keepdim=True)

    patch_locations = torch.zeros(batch_size, 2, dtype=torch.int32)
    for i in range(batch_size):
        patch_locations[i] = torch.tensor(
            unravel_index(
                patch_locations_flat[i],
                (
                    ground_truth.size(2) - patch_size[0],
                    ground_truth.size(3) - patch_size[1],
                ),
            ),
            dtype=torch.int32,
        )

    mask = torch.zeros_like(ground_truth, dtype=torch.bool)

    for i in range(batch_size):
        mask[
            i,
            :,
            patch_locations[i, 0] : patch_locations[i, 0] + patch_size[0],
            patch_locations[i, 1] : patch_locations[i, 1] + patch_size[1],
        ] = True

    return mask
