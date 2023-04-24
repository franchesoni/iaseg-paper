import os
from torch.utils.data import Dataset
import shutil
import json
from pathlib import Path
import numpy as np
import torchvision
import kornia as K
import cv2
from PIL import Image
import pytorch_lightning as pl
import tqdm
from torch.utils.data import DataLoader
from kornia.constants import DataKey, Resample


def tonp_transform(img, mask):
    return np.array(img), np.array(mask)


totensor = torchvision.transforms.ToTensor()
to_01 = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x * 0


def default_dataset_transform(img, mask):
    img = img.resize((896, 896))
    mask = mask.resize((896, 896))
    img, mask = totensor(img), totensor(mask)
    img, mask = to_01(img), to_01(mask)
    return img, mask


def get_img_mask_transform(transform):
    def img_mask_transform(img, mask):
        img = transform(img)
        mask = transform(mask)
        return img, mask

    return img_mask_transform


class NDD20Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        subdataset="below",
        transforms=default_dataset_transform,
        reset=False,
        split="train",
    ):
        assert split in ["train", "val", "test", "all"], "split must be 'train' or 'test' or 'val' or 'all'"
        assert subdataset.upper() in [
            "ABOVE",
            "BELOW",
        ], "subdataset must be 'above' or 'below'"
        self.subdataset = subdataset.upper()
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.imgs_dir = self.root_dir / self.subdataset
        self.masks_dir = self.root_dir / f"{self.subdataset}_MASKS"
        self.split = split
        if reset:
            shutil.rmtree(self.masks_dir)
        if not self.masks_dir.exists():
            annotations_path = self.root_dir / f"{self.subdataset}_LABELS.json"
            with open(annotations_path, "r") as f:
                annotations = json.load(f)
            full_annotation_keys = sorted(annotations.keys())
            self.generate_masks(annotations, full_annotation_keys)
        self.all_masks = sorted(self.masks_dir.glob("*.png"))
        self.all_imgs = sorted(self.imgs_dir.glob("*.jpg"))
        trainval_masks = self.all_masks[: int(len(self.all_masks) * 0.8)]
        trainval_imgs = self.all_imgs[: int(len(self.all_imgs) * 0.8)]
        self.test_masks = self.all_masks[int(len(self.all_masks) * 0.8) :]
        self.test_imgs = self.all_imgs[int(len(self.all_imgs) * 0.8) :]
        if split == "all":
            self.imgs = self.all_imgs
            self.masks = self.all_masks
        elif split == "train":
            self.imgs = trainval_imgs[: int(len(trainval_imgs) * 0.8)]
            self.masks = trainval_masks[: int(len(trainval_masks) * 0.8)]
        elif split == "val":
            self.imgs = trainval_imgs[int(len(trainval_imgs) * 0.8) :]
            self.masks = trainval_masks[int(len(trainval_masks) * 0.8) :]
        elif split == "test":
            self.imgs = self.test_imgs
            self.masks = self.test_masks
        self.check_default_format()

    def generate_masks(self, annotations, annotation_keys):
        self.masks_dir.mkdir(exist_ok=True)
        for ann_key in tqdm.tqdm(annotation_keys):
            img_name = annotations[ann_key]["filename"]
            img = np.array(Image.open(self.imgs_dir / img_name))
            for region in annotations[ann_key]["regions"]:
                polyline = region["shape_attributes"]
                assert (
                    polyline["name"] == "polyline"
                ), "Only polyline regions are supported"
                points = [
                    (x, y)
                    for x, y in zip(polyline["all_points_x"], polyline["all_points_y"])
                ]
                mask = np.zeros_like(img)
                cv2.drawContours(mask, [np.array(points)], 0, (255, 255, 255), -1)
                mask = 0 < mask[..., 0]  # boolean mask
                mask = Image.fromarray(mask)
                mask.save(self.masks_dir / f"mask_{img_name.split('.')[0]}.png")

    def __len__(self):
        return len(self.imgs)

    def get_img_mask(self, idx):
        img = Image.open(self.imgs[idx])
        mask = Image.open(self.masks[idx])
        return img, mask

    def __getitem__(self, idx):
        img, mask = self.get_img_mask(idx)
        H, W = img.height, img.width
        if self.transforms:
            img, mask = self.transforms(img, mask)
        return {"input": img, "target": mask, "hw": (H, W)}

    def convert_to_mmlab(self):
        """Converts dataset to mmlab format"""
        self.check_default_format()
        # create train / val subfolders
        imgs_train_dir = self.imgs_dir / "train"
        imgs_val_dir = self.imgs_dir / "val"
        imgs_train_dir.mkdir(exist_ok=False)
        imgs_val_dir.mkdir(exist_ok=False)
        masks_train_dir = self.masks_dir / "train"
        masks_val_dir = self.masks_dir / "val"
        masks_train_dir.mkdir(exist_ok=False)
        masks_val_dir.mkdir(exist_ok=False)
        # move images and masks to train / val subfolders
        for img, mask in zip(self.train_imgs, self.train_masks):
            shutil.move(img, imgs_train_dir / img.name)
            shutil.move(mask, masks_train_dir / mask.name)
        for img, mask in zip(self.test_imgs, self.test_masks):
            shutil.move(img, imgs_val_dir / img.name)
            shutil.move(mask, masks_val_dir / mask.name)
        return {"seg_map_suffix": "_mask.png"}

    def check_default_format(self):
        # check that format is default, this is, two folders, one for images and one for masks
        assert self.masks_dir.exists(), "Masks directory does not exist"
        assert self.imgs_dir.exists(), "Images directory does not exist"
        # check that folders contain the images and masks
        assert len(list(self.masks_dir.glob("*.png"))) == len(
            list(self.imgs_dir.glob("*.jpg"))
        ), "Number of masks and images is not the same"
        # check that masks are named mask_{img_name}.png
        assert all(
            [mask.stem.split("_")[0] == "mask" for mask in self.masks_dir.glob("*.png")]
        ), "Masks are not named correctly"
        # check that images and masks are paired
        assert [mask.stem.split("_")[-1] for mask in self.all_masks] == [
            img.stem for img in self.all_imgs
        ], "Masks and images are not aligned"

    def convert_from_mmlab(self):
        """Converts back to default format"""
        # move images and masks to root folder
        for img, mask in zip(self.train_imgs, self.train_masks):
            shutil.move(img, self.imgs_dir / img.name)
            shutil.move(mask, self.masks_dir / mask.name)
        for img, mask in zip(self.test_imgs, self.test_masks):
            shutil.move(img, self.imgs_dir / img.name)
            shutil.move(mask, self.masks_dir / mask.name)
        # remove train / val subfolders
        shutil.rmtree(self.imgs_dir / "train")
        shutil.rmtree(self.imgs_dir / "val")
        shutil.rmtree(self.masks_dir / "train")
        shutil.rmtree(self.masks_dir / "val")
        # check everything is ok
        self.check_default_format()


class NDD20DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=42):
        super().__init__()
        self.data_dir = data_dir
        self.default_transform = default_dataset_transform
        self.train_transforms = K.augmentation.AugmentationSequential(
            K.augmentation.RandomResizedCrop((448, 448), same_on_batch=True, align_corners=False),
            K.augmentation.RandomHorizontalFlip(p=0.5, same_on_batch=True),
            K.augmentation.RandomVerticalFlip(p=0.5, same_on_batch=True),
            K.augmentation.RandomRotation(degrees=180, p=0.5, same_on_batch=True),
            K.augmentation.ColorJiggle(p=0.5, same_on_batch=True),
            data_keys=["input", "mask"],
            extra_args={DataKey.MASK: dict(resample=Resample.BILINEAR, align_corners=None)}
        )
        self.batch_size = batch_size

    def prepare_data(self):
        # read `DATASETS_DIR` variable from `segmentation_datasets/ndd20.sh`
        with open("segmentation_datasets/ndd20.sh", "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("DATASETS_DIR"):
                    datasets_dir = line.split("=")[1].strip()[1:-1]  # ignore quotes
                    break
        if (Path(datasets_dir) / "NDD20").exists():
            print("NDD20 dataset already downloaded")
            return
        else:
            # confirm path in file is ok
            input("Hit any key if you've already changed the path in `ndd20.sh`")
            # download by running bash ndd20.sh
            bashCommand = "bash segmentation_datasets/ndd20.sh"
            import subprocess

            process = subprocess.Popen(
                bashCommand.split(), stdout=subprocess.PIPE, cwd="."
            )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.ndd20_train = NDD20Dataset(
                self.data_dir,
                subdataset="below",
                transforms=self.default_transform,
                split="train",
                reset=False,
            )
            self.ndd20_val = NDD20Dataset(
                self.data_dir,
                subdataset="below",
                transforms=self.default_transform,
                split="val",
                reset=False,
            )
        if stage == "test":
            self.ndd20_test = NDD20Dataset(
                self.data_dir,
                subdataset="below",
                transforms=self.default_transform,
                split="test",
                reset=False,
            )
        if stage == "predict":
            self.ndd20_predict = NDD20Dataset(
                self.data_dir,
                subdataset="below",
                transforms=self.default_transform,
                split="test",
                reset=False,
            )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            img, mask = self.train_transforms(batch["input"], batch["target"])
        return {"input": img, "target": mask, "hw": batch["hw"]}

    def train_dataloader(self):
        return DataLoader(
            self.ndd20_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.ndd20_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.ndd20_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count(),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.ndd20_predict,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count(),
        )


def experimentation():
    import matplotlib.pyplot as plt

    ndd20dir = Path("/home/franchesoni/adisk/datasets/NDD20")
    jsonfiles = sorted(ndd20dir.glob("*.json"))
    with open(jsonfiles[0], "r") as f:
        above_labels = json.load(f)

    with open(jsonfiles[1], "r") as f:
        below_labels = json.load(f)
    # *_labels is a dict. The keys are the image name and some strange id.
    # each value is a dict with keys "filename", "size", "file_attributes" and "regions"
    # when using "regions"
    # *_labels[key]["regions"] is a list (with 1 or 2 elements)
    # inside each element of a region we have shape attributes that determine a polyline

    exsample = below_labels[sorted(below_labels.keys())[0]]
    img = Image.open(ndd20dir / "BELOW" / exsample["filename"])
    img.save("img.png")
    polyline = exsample["regions"][0]["shape_attributes"]
    plt.figure()
    plt.imshow(np.array(img))
    plt.plot(polyline["all_points_x"], polyline["all_points_y"], "r")
    plt.savefig("border")

    import cv2

    points = [
        (x, y) for x, y in zip(polyline["all_points_x"], polyline["all_points_y"])
    ]
    mask = np.zeros_like(img)[..., 0]  # 2D
    breakpoint()
    cv2.drawContours(mask, [np.array(points)], 0, (True), -1)
    Image.fromarray(mask).save("mask.png")


def visualize_first_images():
    dataset = NDD20Dataset(
        Path("/home/franchesoni/adisk/datasets/NDD20"), subdataset="below", reset=False
    )
    for ind in range(len(dataset)):
        img, mask = dataset.get_img_mask(ind)
        img.save(f"img_{ind}.png")
        mask.save(f"mask_{ind}.png")
        if ind > 10:
            break


if __name__ == "__main__":
    visualize_first_images()
