import torch

from corrections.mask_generation import random_patch
from corrections.corrections import propagated_correction


def gradient_norm(model):
    """
    Calculate the gradient 2-norm of a model's parameters

    Parameters:
        model: model with parameters to be considered
    """
    total_norm = torch.zeros((1,))

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.pow(2).cpu()

    total_norm = total_norm.sqrt()

    return total_norm.item()


def log_prob_loss(distribution, target):
    """
    Calculates the negative log probability of the target from the distribution (summed over batch)

    Parameters:
        distribution (LowRankMultivariateNormalWrapper): distribution to use
        target (BxCxHxW Tensor): target to calculate log probability of

    Returns:
        result (singleton Tensor)
    """
    return -distribution.td_distribution.log_prob(target.flatten(start_dim=1)).sum()


def lagrangian(primary_loss, *slack_terms, damping=10.0):
    """
    Calculates a lagrangian optimisation loss

    Parameters:
        primary_loss: loss to be optimised when all conditions are met
        *slack_terms: any number of (loss, slack, lagrangian_lambda) tuples or conditions to be met

    Returns:
        result (singleton Tensor)
    """
    slack_terms = list(slack_terms)

    constraint_loss = torch.zeros_like(primary_loss, requires_grad=True)

    for (loss, slack, lagrangian_lambda) in slack_terms:
        damp = damping * (slack - loss).detach()
        constraint_loss = constraint_loss - (lagrangian_lambda - damp) * (slack - loss)

    return primary_loss + constraint_loss
