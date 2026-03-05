"""CIFAR-10 data loading (optional standard augmentation)."""

import io
from contextlib import redirect_stderr, redirect_stdout

import torch
import torchvision
import torchvision.transforms as transforms


def _quiet_cifar10(*, root, train, transform, download):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return torchvision.datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download,
        )


def get_dataloaders(
    train_batch_size: int,
    test_batch_size: int,
    root: str = "~/tmp/cifar10/",
    augmentation: bool = False,
):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tfms = []
    if augmentation:
        train_tfms += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    train_tfms += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    train_transform = transforms.Compose(train_tfms)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = _quiet_cifar10(root=root, train=True,  transform=train_transform, download=True)
    test_ds  = _quiet_cifar10(root=root, train=False, transform=test_transform,  download=True)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=test_batch_size, shuffle=False,
        num_workers=0, drop_last=True,
    )
    return train_dl, test_dl