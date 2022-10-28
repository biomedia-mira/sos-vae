from tqdm import tqdm

import torch
import torch.distributions as td
import torch.nn as nn

from evaluation.loss import (
    log_prob_loss,
    gradient_norm,
    lagrangian,
)
from visualiser.visualiser import visualise_distribution
from visualiser.visualiser import generate_new_results

MAX_NUM_SHOW = 10


def _epoch_core(
    train,
    model,
    loader,
    optimizer,
    device,
    epoch,
    target_kl=None,
    target_ent=None,
    latent_prior=None,
    patch_size=None,
    pre_train=0,
    lagrangian_lambda=None,
    lagrangian_optimizer=None,
):
    """
    Run one epoch

    Paramaters:
        train (bool): whether to train (or to validate)
        model: model to train/validate
        loader (DataLoader): DataLoader to be used
        optimizer: optimizer to be used
        device: device to be used
        epoch (int): current epoch
        target_kl (float): targeted KL loss per batch item
        target_var (float): targeted mean variance of output distribution
        latent_prior (td.distribution): latent prior for use in vae
        patch_size (int): patch_size to be used for correction propagation in the visualiser
        pre_train (int): number of epochs to pre_train for
        lagrangian_lambda (tuple of singleton Tensor): values for lagrangian lambdas
        lagrangian_optimizer (tuple of optimizer): optimizers for lagrangian lambdas

    Returns:
        report (dict): key-value pairs for summary of the epoch
    """
    if patch_size is None:
        raise RuntimeError("patch_size argument not provided")
    if target_kl is None or latent_prior is None:
        raise RuntimeError("target_kl or latent_prior arguments not specified")
    if (lagrangian_lambda is None) != (lagrangian_optimizer is None):
        raise RuntimeError(
            "Both or neither lagrangian_lambda or lagrangian_optimizer should be specified"
        )
    if lagrangian_lambda is not None:
        if (not isinstance(lagrangian_lambda, tuple)) or (
            not isinstance(lagrangian_optimizer, tuple)
        ):
            raise RuntimeError("lagrangian terms should be matched tuples")
        if len(lagrangian_lambda) != len(lagrangian_optimizer):
            raise RuntimeError("lagrangian terms should be matched tuples")

    loss_total = 0.0
    recon_loss_total = 0.0
    kl_loss_total = 0.0
    mean_var_total = 0.0
    mean_entropy_total = 0.0

    for data in tqdm(
        loader,
        total=len(loader),
        desc="Training  " if train else "Validating",
    ):

        img_data = data[0].to(torch.get_default_dtype()).to(device)
        target_data = data[0].to(torch.get_default_dtype()).to(device)

        if train:
            model.train()

            optimizer.zero_grad()
            if lagrangian_lambda is not None:
                for optim in list(lagrangian_optimizer):
                    optim.zero_grad()

        else:
            model.eval()

        distribution = model(img_data, mean_only=epoch < pre_train)

        enc_distribution, distribution = distribution

        recon_loss = log_prob_loss(distribution, target_data)
        kl_loss = td.kl_divergence(enc_distribution, latent_prior).sum()

        if target_ent is not None:
            (ll_kl, ll_ent) = lagrangian_lambda

            entropy = distribution.td_distribution.entropy().sum()

            loss = lagrangian(
                recon_loss,
                (kl_loss, target_kl * img_data.size(0), ll_kl),
                (entropy, target_ent * img_data.size(0), ll_ent),
            )
        else:
            (ll_kl,) = lagrangian_lambda
            loss = lagrangian(
                recon_loss,
                (kl_loss, target_kl * img_data.size(0), ll_kl),
            )

        if train:
            loss.backward()
            optimizer.step()

            if lagrangian_lambda is not None:
                for lam, optim in zip(lagrangian_lambda, lagrangian_optimizer):
                    if lam.grad is not None:
                        lam.grad.neg_()
                        optim.step()

                        # Clamp lagrangian lambda
                        if lam < 0:
                            lam.data = lam.data * 0.0

        mean_var_total += distribution.variance.mean().item()
        mean_entropy_total += distribution.td_distribution.entropy().mean().item()

        # Report mean loss across batch since loss is reduced by sum across batches for generative case
        loss_total += loss.item() / img_data.size(0)
        recon_loss_total += recon_loss.item() / img_data.size(0)
        kl_loss_total += kl_loss.item() / img_data.size(0)

    report = dict()
    report["Loss"] = loss_total / len(loader)
    report["Mean Variance"] = mean_var_total / len(loader)
    report["Mean Entropy"] = mean_entropy_total / len(loader)
    if train:
        report["Gradient Norm"] = gradient_norm(model)
    report["Variance"] = distribution.variance
    report["Reconstructions/Predictions"] = visualise_distribution(
        distribution,
        img_data,
        min(distribution.batch_size, MAX_NUM_SHOW),
        patch_size=patch_size,
        title="Reconstructions/Predictions",
    )
    if lagrangian_lambda is not None:
        i = 0
        for lam in list(lagrangian_lambda):
            i += 1
            report[f"Lagrangian Lambda {i}"] = lam

    num_show = min(distribution.batch_size, MAX_NUM_SHOW)
    report["Latent Samples"] = generate_new_results(
        model,
        latent_prior,
        num_show,
        tuple(img_data.shape)[1:],
        title="Reconstructions/Predictions",
        mean_only=epoch < pre_train,
    )
    report["KL Loss"] = kl_loss_total / len(loader)
    report["Reconstruction Loss"] = recon_loss_total / len(loader)

    return report


def run_epoch(train, *args, **kwargs):
    if train:
        return _epoch_core(train, *args, **kwargs)

    with torch.no_grad():
        return _epoch_core(train, *args, **kwargs)
