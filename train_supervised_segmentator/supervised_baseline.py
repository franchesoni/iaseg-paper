import pytorch_lightning as pl
from pathlib import Path
import torchvision
import torch

from segmentation_datasets.ndd20 import NDD20Dataset
from config import DATA_DIR, SEED

class Segmentator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)
        self.model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))  # 1 class
        self.loss = torchvision.ops.sigmoid_focal_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)['out']
        loss = self.loss(y_hat, y, reduction='mean')
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def train_segmentator():
    torch.manual_seed(SEED)
    segmentator = Segmentator()
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((448, 448)),
        torchvision.transforms.ToTensor(),
    ])
    def img_mask_transform(img, mask):
        img = transforms(img)
        mask = transforms(mask)
        return img, mask
        
    dataset = NDD20Dataset(root_dir=Path(DATA_DIR) / "NDD20", subdataset='below', split='train', transform=img_mask_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, drop_last=True) 
    trainer = pl.Trainer(accelerator="gpu", max_epochs=1, limit_train_batches=4, fast_dev_run=True)
    trainer.fit(segmentator, dataloader)

    breakpoint()

train_segmentator()