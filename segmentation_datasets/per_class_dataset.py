from pathlib import Path
from PIL import Image
import numpy as np
import torch

class PerClassDataset(torch.utils.data.Dataset):
  def __init__(self, input_dir, target_dir, label):
    super().__init__()
    self.input_dir = input_dir
    self.target_dir = target_dir
    self.label = label

    self.input_files = sorted(Path(input_dir).glob('*.png'))
    self.target_files = sorted(Path(target_dir).glob(f'*_{label:02d}.png'))
    assert len(self.input_files) == len(self.target_files)

  def __len__(self):
    return len(self.input_files)

  def __getitem__(self, index):
    input_file = self.input_files[index]
    target_file = self.target_files[index]
    input = Image.open(input_file).convert("RGB")  # for if it has alpha channel
    target = Image.open(target_file)
    return input, target

