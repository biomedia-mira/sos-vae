import argparse
import os

# N.B: python random should not be used for anything reproducible as it is not seeded. Use here is for ID generation
import random

import torch
import torch.distributions as td
from torch.utils.tensorboard import SummaryWriter

import matplotlib

from datasets.data_loaders import load_celeba_data

from model.vae import VAE

from train.train import run_epoch

# Argument parsing
parser = argparse.ArgumentParser(description="Train and evaluate network")

parser.add_argument(
    "-g", "--gpu", action="store_true", help="Evaluate using GPU if available"
)
parser.add_argument(
    "-f64",
    "--float64",
    action="store_true",
    help="Evaluate using 64-bit floats. 32 otherwise",
)
parser.add_argument(
    "-dnv",
    "--dist-no-val",
    action="store_true",
    help="Disable distribution validation checks",
)
parser.add_argument(
    "-d",
    "--deterministic",
    action="store_true",
    help="Set deterministic/repeatable computation",
)
parser.add_argument(
    "-t",
    "--title",
    action="store_true",
    help="Enable title prompt",
)
parser.add_argument(
    "-s",
    "--save",
    action="store_true",
    help="Save model_dict upon run completion",
)
parser.add_argument(
    "-e", "--epochs", type=int, help="Number of epochs to train for", default=10
)
parser.add_argument("-b", "--batch-size", type=int, help="Batch size", default=25)
parser.add_argument(
    "-lr", "--learning-rate", type=float, help="Learning rate", default=1e-4
)
parser.add_argument(
    "-f",
    "--latent-features",
    type=int,
    help="Dimensionality of latent space",
    default=16,
)
parser.add_argument(
    "-r",
    "--rank",
    type=int,
    help="Rank of the decoder covariance",
    default=5,
)
parser.add_argument(
    "-kl",
    "--target-kl",
    type=float,
    help="Target KL loss (per batch item)",
    default=45.0,
)
parser.add_argument(
    "-ent",
    "--target-ent",
    type=float,
    help="Target entropy of the output distribution",
    default=None,
)
parser.add_argument(
    "-p",
    "--patch-size",
    type=int,
    help="Size of the ground truth patch used for correction in the visualiser",
    default=10,
)
parser.add_argument(
    "-pre",
    "--pre-train",
    type=int,
    help="Number of epochs to pre-train the model for",
    default=0,
)
parser.add_argument(
    "--grayscale",
    action="store_true",
    help="Run in grayscale",
)
parser.add_argument(
    "-cp",
    "--checkpoints",
    type=int,
    help="Epoch interval between checkpoints. No checkpoints otherwise",
)
parser.add_argument(
    "--resume",
    type=int,
    help="ID of a previous run to resume running (if specified). Resuming with modified args should be attempted with caution",
)

args = parser.parse_args()


def new_id():
    chosen_id = None
    while chosen_id is None or os.path.isdir(f"out/{chosen_id}"):
        chosen_id = random.randint(1e3, 1e4)
    return chosen_id


def latest_checkpoint(run_id):
    directory = f"out/{run_id}"

    max_valid = -1
    for filename in os.listdir(directory):
        if filename.endswith("-checkpoint.tar"):
            cur_id = int(filename[: -len("-checkpoint.tar")])
            max_valid = max(max_valid, cur_id)

    if max_valid <= 0:
        raise RuntimeError(f"No valid checkpoints saved for ID: {run_id}")
    return torch.load(f"{directory}/{max_valid}-checkpoint.tar")


# Define constants
if args.gpu and not torch.cuda.is_available():
    print("[WARNING]\t GPU evaluation requested and no CUDA device found: using CPU")
DEVICE = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

if args.float64:
    torch.set_default_dtype(torch.float64)

if args.dist_no_val:
    td.Distribution.set_default_validate_args(False)

if args.deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(100)  # 42

CHECKPOINTS = args.checkpoints
RESUME = args.resume is not None
SAVE = args.save

if RESUME:
    ID = args.resume
    print(f"Resuming ID: {ID}")
elif SAVE or CHECKPOINTS is not None:
    ID = new_id()
    os.mkdir(f"out/{ID}")
    print(f"Saving with ID: {ID}")

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
LATENT_FEATURES = args.latent_features
DECODER_RANK = args.rank
TARGET_KL = args.target_kl
TARGET_ENT = args.target_ent
PATCH_SIZE = args.patch_size
PRE_TRAIN_EPOCHS = args.pre_train
ASK_TITLE = args.title
CHANNELS = 1 if args.grayscale else 3


def tensorboard_write(writer, key, value, prefix):
    if isinstance(value, matplotlib.figure.Figure):
        writer.add_figure(f"{prefix} {key}", value, global_step=epoch + 1)
    elif isinstance(value, float):
        writer.add_scalar(f"{prefix} {key}", value, global_step=epoch + 1)
    elif isinstance(value, torch.Tensor):
        if len(value.shape) == 0 or (len(value.shape) == 1 and value.size(0) == 1):
            writer.add_scalar(f"{prefix} {key}", value, global_step=epoch + 1)
        else:
            writer.add_histogram(f"{prefix} {key}", value, global_step=epoch + 1)
    else:
        raise RuntimeError(f"Unknown type to write to tensorboard: {type(value)}")


# Main method
if __name__ == "__main__":
    starting_epoch = 1
    if RESUME:
        if not os.path.isdir(f"out/{ID}"):
            raise RuntimeError(f"No checkpoint folder: {ID}")
        checkpoint = latest_checkpoint(ID)
        starting_epoch = checkpoint["epoch"] + 1

    # Create data loaders
    (training_loader, validation_loader, test_loader) = load_celeba_data(
        BATCH_SIZE, args.grayscale
    )

    model = VAE(LATENT_FEATURES, DECODER_RANK, CHANNELS).to(DEVICE)

    latent_prior = td.MultivariateNormal(
        torch.zeros(LATENT_FEATURES).to(DEVICE),
        torch.eye(LATENT_FEATURES).to(DEVICE),
    )

    # Set up optimsier
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Set up lagrangian lambdas
    lagrangian_lambda = torch.zeros(1, requires_grad=True, device=DEVICE)
    ll_optimizer = torch.optim.SGD([lagrangian_lambda], lr=1.0)

    lagrangian_lambda2 = torch.zeros(1, requires_grad=True, device=DEVICE)
    ll_optimizer2 = torch.optim.SGD([lagrangian_lambda2], lr=1.0)

    # Set up logging
    writer = SummaryWriter()

    if ASK_TITLE:
        writer.add_text("Title", input("Title of run: "))

    if (SAVE or CHECKPOINTS is not None) and not RESUME:
        writer.add_text("ID", f"{ID}")

    if RESUME:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        lagrangian_lambda.data = checkpoint["lagrangian_lambda"].data
        ll_optimizer.load_state_dict(checkpoint["ll_optim_state"])
        lagrangian_lambda2.data = checkpoint["lagrangian_lambda2"].data
        ll_optimizer2.load_state_dict(checkpoint["ll_optim2_state"])
        torch.set_rng_state(checkpoint["rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    for epoch in range(starting_epoch - 1, EPOCHS):
        print(f"Epoch {epoch+1} of {EPOCHS}")

        train_report = run_epoch(
            True,
            model,
            training_loader,
            optimizer,
            DEVICE,
            epoch,
            target_kl=TARGET_KL,
            target_ent=TARGET_ENT,
            latent_prior=latent_prior,
            patch_size=PATCH_SIZE,
            pre_train=PRE_TRAIN_EPOCHS,
            lagrangian_lambda=(lagrangian_lambda,)
            if TARGET_ENT is None
            else (lagrangian_lambda, lagrangian_lambda2),
            lagrangian_optimizer=(ll_optimizer,)
            if TARGET_ENT is None
            else (ll_optimizer, ll_optimizer2),
        )
        validation_report = run_epoch(
            False,
            model,
            validation_loader,
            optimizer,
            DEVICE,
            epoch,
            target_kl=TARGET_KL,
            target_ent=TARGET_ENT,
            latent_prior=latent_prior,
            patch_size=PATCH_SIZE,
            pre_train=PRE_TRAIN_EPOCHS,
            lagrangian_lambda=(lagrangian_lambda,)
            if TARGET_ENT is None
            else (lagrangian_lambda, lagrangian_lambda2),
            lagrangian_optimizer=(ll_optimizer,)
            if TARGET_ENT is None
            else (ll_optimizer, ll_optimizer2),
        )

        for key in train_report.keys():
            tensorboard_write(writer, key, train_report[key], "Train")

        for key in validation_report.keys():
            tensorboard_write(writer, key, validation_report[key], "Validation")

        if CHECKPOINTS is not None and (epoch + 1) % CHECKPOINTS == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "lagrangian_lambda": lagrangian_lambda,
                "ll_optim_state": ll_optimizer.state_dict(),
                "lagrangian_lambda2": lagrangian_lambda2,
                "ll_optim2_state": ll_optimizer2.state_dict(),
            }
            torch.save(checkpoint, f"out/{ID}/{epoch+1}-checkpoint.tar")

    writer.close()

    if SAVE:
        torch.save(model.state_dict(), f"out/{ID}/model.pt")
