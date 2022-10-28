import torch
from torch.utils.data import DataLoader, random_split

from torchvision import datasets
from torchvision import transforms


def celeba_atrib_to_label(atrib):
    """
    Extract attributes from an attribute tensor to a label
    """

    # Extract "Male" attribute
    return atrib[:, 20].int()


def load_celeba_data(batch_size, grayscale):
    """
    Create the data loaders for the CELEBA dataset.

    Parameters:
        batch_size (int): the batch size to use
        grayscale (bool): whether to load grayscale images

    Returns:
        tuple of (training_loader, validation_loader)
    """
    # Data pre-processing transformation
    trans = list()
    if grayscale:
        trans.append(transforms.Grayscale())
    trans.append(transforms.RandomHorizontalFlip())
    trans.append(transforms.ToTensor())

    pre_proc_trans = transforms.Compose(trans)

    img_dataset = datasets.CelebA(
        "./data/CELEBA-torchvision",
        split="all",
        transform=pre_proc_trans,
        target_type="attr",
    )

    # Randomly shrink dataset (for testing)
    train_num = 10000
    train_num = min(train_num, len(img_dataset))

    img_dataset, _ = random_split(
        img_dataset,
        [train_num, len(img_dataset) - train_num],
        generator=torch.Generator().manual_seed(42),
    )

    n_val = int(len(img_dataset) * 0.2)
    n_test = int(len(img_dataset) * 0.1)
    n_train = len(img_dataset) - n_val - n_test
    training_set, validation_set, test_set = random_split(
        img_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return training_loader, validation_loader, test_loader


def load_brain_thumbnail_data(batch_size, grayscale):
    """
    Create the data loaders for the brain thumbnail dataset

    Parameters:
        batch_size (int): the batch size to use
        grayscale (bool): whether to load grayscale images

    Returns:
        tuple of (training_loader, validation_loader)
    """
    # Data pre-processing transformation
    trans = list()
    if grayscale:
        trans.append(transforms.Grayscale())
    trans.append(transforms.RandomHorizontalFlip())
    trans.append(transforms.ToTensor())

    pre_proc_trans = transforms.Compose(trans)

    img_dataset = datasets.ImageFolder("./data/BRAINS", transform=pre_proc_trans)

    # Randomly shrink dataset (for testing)
    train_num = 10000
    train_num = min(train_num, len(img_dataset))

    img_dataset, _ = random_split(
        img_dataset,
        [train_num, len(img_dataset) - train_num],
        generator=torch.Generator().manual_seed(42),
    )

    n_val = int(len(img_dataset) * 0.2)
    n_test = int(len(img_dataset) * 0.1)
    n_train = len(img_dataset) - n_val - n_test
    training_set, validation_set, test_set = random_split(
        img_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return training_loader, validation_loader, test_loader
