import os
from pathlib import Path
import time

import psutil
import torch
import tqdm

from continuously_train import ContinuousDataset
from config import DATA_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num_cpus = len(psutil.Process().cpu_affinity())
label = 4

data_dir = os.path.join(DATA_DIR, "perlabel_sbd")
dataset = ContinuousDataset(
        Path(data_dir) / "inputs", Path(data_dir) / "annotations", label
    )

dataset.update_file_list()  # the reason why to use plain pytorch
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_cpus,
    drop_last=True,
    pin_memory=True,
    # prefetch_factor=4,
)

last = time.time()

with tqdm.tqdm(enumerate(dataloader)) as pbar:
    for i, (img, target) in pbar:
        img, target = img.to(DEVICE), target.to(DEVICE)
        pass

