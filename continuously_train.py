import os
import sys
import random
from pathlib import Path
import time
import tqdm
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

from config import DATA_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def debug_id(x, **kwargs):
  print(x.shape)
  breakpoint()
  return x

class ContinuousDataset(torch.utils.data.Dataset):
  def __init__(self, input_dir, annotations_dir, label, min_length=32):
    self.input_dir = Path(input_dir)
    self.annotations_dir = Path(annotations_dir)
    self.label = label

    self.min_length = min_length
    self.update_file_list()  # receives min_length
    self.aug = A.Compose([  # we could add more augmentations here
        # A.Lambda(image=debug_id, mask=debug_id),
        A.OneOf([
          A.PadIfNeeded(min_height=480, min_width=480),
          A.PadIfNeeded(min_height=960, min_width=960, p=0.2),
        ], p=1),
        A.RandomSizedCrop(min_max_height=(88, 480), height=480, width=480),  # sbd params
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ElasticTransform(),
            A.GridDistortion(),
            A.OpticalDistortion()                  
            ], p=0.3),
        A.OneOf([
          A.CLAHE(),
          A.RandomBrightnessContrast(),    
          A.RandomGamma()], p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])

  def update_file_list(self):
    img_indices = [p.stem.split('_')[1] for p in self.input_dir.glob("img_*.png")]
    annotation_indices = [ann.name.split('_')[1] for ann in self.annotations_dir.glob(f"annotation_*_{str(self.label).zfill(2)}.png")]
    self.common_indices = sorted(list(set(img_indices).intersection(set(annotation_indices))))
    if len(self.common_indices):  # shouldn't be 0
      self.common_indices = self.common_indices * (self.min_length // len(self.common_indices) + 1)
    return self.common_indices

  def __len__(self):
    return len(self.common_indices)

  def __getitem__(self, idx):
    image = np.array(Image.open(self.input_dir / f"img_{self.common_indices[idx]}.png"))
    mask = (np.array(Image.open(self.annotations_dir / f"annotation_{self.common_indices[idx]}_{str(self.label).zfill(2)}.png"))!=0).astype(np.uint8)
    augmented = self.aug(image=image, mask=mask)
    return augmented['image'], augmented['mask']

def launch_continuously_train(data_dir, label):
  seed_everything()
  model = smp.Unet(
    # encoder_name = "resnet18",
    encoder_name="timm-mobilenetv3_small_minimal_100",
    encoder_weights = "imagenet",
    in_channels = 3,
    classes = 1,
    activation = "sigmoid",
  ).to(DEVICE)
  # model = smp.Unet(
  #   encoder_name = "tu-hrnet_w18",
  #   encoder_weights = "imagenet",
  #   in_channels = 3,
  #   classes = 1,
  #   activation = "sigmoid",
  # )

  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  dataset = ContinuousDataset(Path(data_dir) / 'inputs', Path(data_dir) / 'annotations', label)
  if len(dataset) < 32:
    print('Waiting for annotations...')
    for wait in tqdm.tqdm(range(1000)):
      time.sleep(1)
      dataset.update_file_list()
      if len(dataset) >= 32:
        break
    if len(dataset) < 32:
      raise RuntimeError('Waited too long for annotations to start training')
  print('Starting training...')
  Path('checkpoints').mkdir(exist_ok=True)

  writer = SummaryWriter()

  st = time.time()
  last = st
  times = {'dataloader': 0, 'loading_data': 0, 'forward': 0, 'backward': 0, 'logging': 0, 'image_logging': 0, 'model_saving': 0}
  iterations = 0
  for epoch in range(10):
    dataset.update_file_list()  # the reason why to use plain pytorch
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, drop_last=True)

    times['dataloader'] += time.time() - last
    last = time.time()

    with tqdm.tqdm(enumerate(dataloader)) as pbar:
      for i, (img, target) in pbar:
        img, target = img.to(DEVICE), target.to(DEVICE)

        times['loading_data'] += time.time() - last
        last = time.time()

        out = model(img)
        loss = torchvision.ops.sigmoid_focal_loss(out, target[:, None, :, :].float(), reduction='mean')

        times['forward'] += time.time() - last
        last = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        times['backward'] += time.time() - last
        last = time.time()

        pbar.set_description(f"Epoch {epoch}, batch {i}, loss {loss.item()}")
        writer.add_scalar('Loss/train', loss.item(), global_step=iterations)
        iterations += 1

        times['logging'] += time.time() - last
        last = time.time()

      writer.add_image('Image/train/input', img[0], epoch)
      writer.add_image('Image/train/output', out[0], epoch)

      times['image_logging'] += time.time() - last
      last = time.time()

    print(f"finished epoch {epoch} with times {times}")
    torch.save(model, f"checkpoints/model_{str(epoch).zfill(4)}.pth")

    times['model_saving'] += time.time() - last
    last = time.time()

    
  
  writer.flush()
  writer.close()
   


if __name__ == '__main__':
  label = int(sys.argv[1])
  data_dir = os.path.join(DATA_DIR, 'perlabel_sbd')
  launch_continuously_train(data_dir, label)

  
