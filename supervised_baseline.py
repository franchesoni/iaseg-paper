import pytorch_lightning as pl
from pathlib import Path
import torchvision
import torch

from segmentation_datasets.ndd20 import NDD20Dataset
from segmentation_datasets.ndd20 import transforms
from config import DATA_DIR, SEED


class Segmentator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
            progress=True,
        )
        self.model.classifier[4] = torch.nn.Conv2d(
            256, 1, kernel_size=(1, 1), stride=(1, 1)
        )  # 1 class
        breakpoint()
        self.loss = torchvision.ops.sigmoid_focal_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)["out"]
        loss = self.loss(y_hat, y, reduction="mean")
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)["out"]
        loss = self.loss(y_hat, y, reduction="mean")
        self.log("test_loss", loss)
        mse = torch.nn.MSELoss()(y_hat, y)
        self.log("test_mse", mse)
        mae = torch.nn.L1Loss()(y_hat, y)
        self.log("test_mae", mae)
        return loss


def train_segmentator():
    torch.manual_seed(SEED)
    segmentator = Segmentator()
    dataset = NDD20Dataset(
        root_dir=Path(DATA_DIR) / "NDD20", subdataset="below", split="train"
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=42,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=48,
        fast_dev_run=False,
        devices=1,
        profiler="simple",
    )
    trainer.fit(segmentator, dataloader)


def evaluate_segmentator():
    torch.manual_seed(SEED)
    segmentator = Segmentator.load_from_checkpoint(
        checkpoint_path="lightning_logs/version_1/checkpoints/epoch=47-step=1968.ckpt"
    )
    dataset = NDD20Dataset(
        root_dir=Path(DATA_DIR) / "NDD20", subdataset="below", split="test"
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=42,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=48,
        fast_dev_run=False,
        devices=1,
        profiler="simple",
    )
    trainer.test(segmentator, dataloader)


def generate_predictions():
    torch.manual_seed(SEED)
    segmentator = Segmentator.load_from_checkpoint(
        checkpoint_path="lightning_logs/version_1/checkpoints/epoch=47-step=1968.ckpt"
    )
    dataset = NDD20Dataset(
        root_dir=Path(DATA_DIR) / "NDD20",
        subdataset="below",
        split="train",
        transform=None,
    )
    segmentator.model.eval()
    with torch.no_grad():
        for ind, (original_img, _) in enumerate(dataset):
            resized_img = transforms(original_img)
            prediction = torch.sigmoid(
                segmentator.model(resized_img[None])["out"][0][0]
            )
            torchvision.utils.save_image(prediction, f"predictions/{ind}.png")


# command to run the script and log output of terminal to a file
# python supervised_baseline.py | tee supervised_baseline.log

if __name__ == "__main__":
    train_segmentator()
    # evaluate_segmentator()
    # generate_predictions()
