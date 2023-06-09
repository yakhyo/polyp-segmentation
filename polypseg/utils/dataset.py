import os
import cv2
import numpy as np

from PIL import Image, ImageOps
from torch.utils import data
from polypseg.utils.general import Augmentation


class PolypDataset(data.Dataset):
    def __init__(
            self,
            root: str,
            filenames: list[str],
            image_size: int = 512,
            transforms: Augmentation = Augmentation(),
            mask_suffix: str = "_mask"
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.mask_suffix = mask_suffix
        self.filenames = filenames
        if not self.filenames:
            raise FileNotFoundError(f"Files not found in {root}")
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # image path
        image_path = os.path.join(self.root, f"images{os.sep}{filename.strip()}.jpg")
        mask_path = os.path.join(self.root, f"masks{os.sep}{filename.strip() + self.mask_suffix}.jpg")

        # image load
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # resize
        image, mask = self.resize_pil(image, mask, image_size=self.image_size)

        _, mask = cv2.threshold(np.array(mask), 127, 255, cv2.THRESH_BINARY)
        mask = Image.fromarray(np.asarray(mask / 255.0, dtype=float))

        assert image.size == mask.size, f"`image`: {image.size} and `mask`: {mask.size} are not the same"

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask

    @staticmethod
    def resize_pil(image, mask, image_size):
        w, h = image.size
        scale = min(image_size / w, image_size / h)

        # resize image
        image = image.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)
        mask = mask.resize((int(w * scale), int(h * scale)), resample=Image.NEAREST)

        # pad size
        delta_w = image_size - int(w * scale)
        delta_h = image_size - int(h * scale)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # pad image
        image = ImageOps.expand(image, (left, top, right, bottom))
        mask = ImageOps.expand(mask, (left, top, right, bottom))

        return image, mask
