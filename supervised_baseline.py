import tqdm
import pytorch_lightning as pl
from pathlib import Path
import torchvision
import torch

from segmentation_datasets.ndd20 import NDD20DataModule
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
        self.loss = torchvision.ops.sigmoid_focal_loss

    def training_step(self, batch, batch_idx):
        x, y, hw = batch["input"], batch["target"], batch["hw"]
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


def run_segmentator(mode='train'):
    torch.manual_seed(SEED)
    datamodule = NDD20DataModule(
        data_dir=Path(DATA_DIR) / "NDD20", 
        batch_size=32,
    )
    if mode in ['train', 'test']:
        segmentator = Segmentator()
        trainer = pl.Trainer(
            log_every_n_steps=1,
            accelerator="gpu",
            max_epochs=24,
            fast_dev_run=False,
            devices=[1],
            profiler="simple",
        )
        if mode == 'train':
            trainer.fit(segmentator, datamodule)
        elif mode == 'test':
            trainer.test(segmentator, datamodule)
    elif mode == 'generate':
        Path('predictions').mkdir(exist_ok=True)
        datamodule.setup('fit')
        segmentator = Segmentator.load_from_checkpoint(
                checkpoint_path=str(list(Path("lightning_logs/version_5/checkpoints/").glob("*.ckpt"))[0])
            )
        segmentator.model.eval()

        with torch.no_grad():
            for ind, (sampledict) in tqdm.tqdm(enumerate(datamodule.ndd20_train)):
                img = sampledict["input"]
                h, w = sampledict["hw"]
                prediction = torch.sigmoid(
                    segmentator.model(img[None])["out"][0][0]
                )
                # resize image back to (h, w)
                prediction = torch.nn.functional.interpolate(
                    prediction[None, None], size=(h, w)
                )[0, 0]
                torchvision.utils.save_image(prediction, f"predictions/{str(ind).zfill(5)}.png")

            indoffset = len(datamodule.ndd20_train)
            for ind, (sampledict) in tqdm.tqdm(enumerate(datamodule.ndd20_val)):
                img = sampledict["input"]
                h, w = sampledict["hw"]
                prediction = torch.sigmoid(
                    segmentator.model(img[None])["out"][0][0]
                )
                prediction = torch.nn.functional.interpolate(
                    prediction[None, None], size=(h, w)
                )[0, 0]
                torchvision.utils.save_image(prediction, f"predictions/{str(ind+indoffset).zfill(5)}.png")

# command to run the script and log output of terminal to a file
# python supervised_baseline.py | tee supervised_baseline.log
if __name__ == "__main__":
    run_segmentator('generate')
