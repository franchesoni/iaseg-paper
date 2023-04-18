import os

from PIL import Image
import numpy as np
import torchvision

from config import DATA_DIR, DOWNLOAD_IF_NEEDED


def get_sbd_dataset():
    download = DOWNLOAD_IF_NEEDED and not os.path.exists(os.path.join(DATA_DIR, "sbd"))
    return torchvision.datasets.SBDataset(
        os.path.join(DATA_DIR, "sbd"),
        image_set="train",
        mode="segmentation",
        download=download,
        transforms=None,
    )


if __name__ == "__main__":
    dataset = get_sbd_dataset()
    for img, mask in dataset:
        img.save("img.png")
        mask = np.array(mask)
        Image.fromarray(((mask / mask.max()) * 255).astype(np.uint8)).save("mask.png")
        breakpoint()
