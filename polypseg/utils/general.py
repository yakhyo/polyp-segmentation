import os
import random

import numpy as np
import torch
from torchvision.transforms import functional as F


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def strip_optimizers(f: str):
    """Strip optimizer from 'f' to finalize training"""
    x = torch.load(f, map_location="cpu")
    for k in "optimizer", "best_score":
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, f)
    mb = os.path.getsize(f) / 1e6  # get file size
    print(f"Optimizer stripped from {f}, saved as {f} {mb:.1f}MB")


class Augmentation:
    """Standard Augmentation"""

    def __init__(self, hflip_prop: float = 0.5) -> None:
        transforms = []
        if hflip_prop > 0:
            transforms.append(RandomHorizontalFlip(hflip_prop))
        transforms.extend([PILToTensor(), ConvertImageDtype(torch.float)])
        self.transforms = Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class PILToTensor:
    """Convert PIL image to torch tensor"""

    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    """Convert Image dtype"""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Compose:
    """Composing all transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip"""

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class EarlyStopping:
    """EarlyStopping"""

    def __init__(self, patience=10):
        self.best_fitness = 0.0  # i.e. mAP, Dice
        self.best_epoch = 0
        self.patience = patience or float('inf')
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            print(f"Early stopping as no improvement observed in last {self.patience} epochs.")
        return stop
