from pathlib import Path
import tqdm
from PIL import Image
import numpy as np
import os
import skimage

import torchvision

from config import DATA_DIR, DOWNLOAD_IF_NEEDED


def find_value_of_closest_pixel_with_different_value(image, x, y):
  value = image[x, y]
  if x != 0:
    if image[x - 1, y] != value:
      return image[x - 1, y]
  if x != image.shape[0] - 1:
    if image[x + 1, y] != value:
      return image[x + 1, y]
  if y != 0:
    if image[x, y - 1] != value:
      return image[x, y - 1]
  if y != image.shape[1] - 1:
    if image[x, y + 1] != value:
      return image[x, y + 1]
  return value
 
def replace_pixel_value(image, value, showprocess=False):
  if showprocess:
    destdir = Path("steps")
    # remove all files in destdir
    for f in destdir.iterdir():
      f.unlink()
    # save the original image
    Image.fromarray(image).save(destdir / "original.png")
    stepcount = 0
  newimage = image.copy()
  while np.any(image == value):
    xs, ys = np.nonzero(image == value)
    for i in range(len(xs)):
      x, y = xs[i], ys[i]
      newimage[x, y] = find_value_of_closest_pixel_with_different_value(image, x, y)
    image = newimage.copy()
    if showprocess:
      Image.fromarray(newimage).save(destdir / f"step_{stepcount}.png")
      stepcount += 1
  return image

def normalize_image(image):
  imin, imax = np.min(image), np.max(image)
  if imin == imax:
    return np.zeros_like(image).astype(np.uint8)
  else:
    image = (image - imin) / (imax - imin)
    return (image * 255).astype(np.uint8)


def generate_voc_masks():
  """Generates masks from the VOC dataset by replacing the borders with the closest label and saving the relabeled version as .png"""
  download = DOWNLOAD_IF_NEEDED and not os.path.exists(os.path.join(DATA_DIR, 'voc'))
  voc_dataset = torchvision.datasets.VOCSegmentation(os.path.join(DATA_DIR, 'voc'), year='2012', image_set="train", download=download, transforms=None)

  voc_masks_dir = Path(DATA_DIR) / "voc_masks" 
  voc_masks_dir.mkdir(exist_ok=True)
  for i, (image, target) in tqdm.tqdm(enumerate(voc_dataset)):
    target = np.array(target)
    target = replace_pixel_value(target, 255)
    # target = skimage.segmentation.relabel_sequential(target)[0]  # relabel everything but label 0
    # target = normalize_image(target)
    # Image.fromarray(target).save(voc_masks_dir / f"mask_{i}.png")


