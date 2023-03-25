import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
from segmentation_datasets.sbd import get_sbd_dataset

def generate_sbd_per_label(input_dir, target_dir):
  sbd_dataset = get_sbd_dataset()
  targetDir = Path(target_dir)
  inputDir = Path(input_dir)
  targetDir.mkdir(exist_ok=True, parents=True)
  inputDir.mkdir(exist_ok=True, parents=True)
  total_len = len(sbd_dataset)
  for i, (img, target) in tqdm.tqdm(enumerate(sbd_dataset), total=total_len):
    img.save(inputDir / f"img_{str(i).zfill(6)}.png")
    nptarget = np.array(target)
    for label in range(21):
      Image.fromarray(nptarget==label).save(targetDir / f"target_{str(i).zfill(6)}_{str(label).zfill(2)}.png")


if __name__ == "__main__":
  generate_sbd_per_label('data/perlabel_sbd/inputs', 'data/perlabel_sbd/targets')