import os

from tqdm import tqdm

import torch
import torch.distributions as td
from torchvision.utils import save_image

from model.vae import VAE

torch.autograd.set_grad_enabled(False)

if __name__ == "__main__":
    SAMPLES = True
    NUM_SAVE = 50000

    OUT_DIR = "samples"
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    GRAYSCALE = False
    SHAPE = (1 if GRAYSCALE else 3, 218, 178)
    RANK = 25
    LATENT_FEATURES = 128

    DEVICE = "cuda"
    BATCH_SIZE = 50

    model = VAE(LATENT_FEATURES, RANK, 1 if GRAYSCALE else 3)
    model.load_state_dict(torch.load("out/model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    latent_prior = td.MultivariateNormal(
        torch.zeros(LATENT_FEATURES, device=DEVICE),
        torch.eye(LATENT_FEATURES, device=DEVICE),
    )

    for b in tqdm(range(0, NUM_SAVE, BATCH_SIZE)):
        latent_samples = latent_prior.sample((BATCH_SIZE,))

        distribution = model.decode(latent_samples)

        means = distribution.unflattened_mean(SHAPE).cpu()
        samples = distribution.td_distribution.sample().unflatten(1, SHAPE).cpu()
        imgs = samples if SAMPLES else means

        for i in range(min(BATCH_SIZE, NUM_SAVE - b)):
            filepath = f"{OUT_DIR}/image-{b+i+1}.png"
            if not os.path.isfile(filepath):
                img = imgs[i]
                save_image(img, filepath)
