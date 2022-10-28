"""
Script to perform iterative PCA on the CELEBA dataset for comparison with distribution-only performance.
"""

import torch

from torchvision.utils import make_grid, save_image

from tqdm import tqdm

from sklearn.decomposition import IncrementalPCA

from datasets.data_loaders import load_celeba_data

BATCH_SIZE = 100
RANK = 25

IMG_SHAPE = (3, 218, 178)


if __name__ == "__main__":
    torch.autograd.set_grad_enabled(False)

    training_loader, validation_loader, test_loader = load_celeba_data(
        BATCH_SIZE, False
    )

    ipca = IncrementalPCA(n_components=RANK, batch_size=BATCH_SIZE)

    for data in tqdm(training_loader, total=len(training_loader)):
        imgs = data[0].flatten(start_dim=1).cpu().numpy()
        ipca.partial_fit(imgs)

    test_imgs = (
        torch.randn((BATCH_SIZE, RANK))
        * torch.from_numpy(ipca.explained_variance_).sqrt()
    )

    test_imgs = ipca.inverse_transform(test_imgs.numpy())

    test_imgs = torch.from_numpy(test_imgs).unflatten(1, IMG_SHAPE)

    fig = make_grid(test_imgs, nrow=10)
    save_image(fig, "ipca.png")
