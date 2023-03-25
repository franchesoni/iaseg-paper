import os

import torchvision

from config import DATA_DIR, DOWNLOAD_IF_NEEDED

def get_sbd_dataset():
  download = DOWNLOAD_IF_NEEDED and not os.path.exists(os.path.join(DATA_DIR, 'sbd'))
  return torchvision.datasets.SBDataset(os.path.join(DATA_DIR, 'sbd'), image_set="train", mode="segmentation", download=download, transforms=None)

