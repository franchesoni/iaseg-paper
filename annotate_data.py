import os
import datetime
import sys
from pathlib import Path
import time
from PIL import Image
import numpy as np
import torchvision
import torch

from segmentation_datasets.per_class_dataset import PerClassDataset
from app.ClickSEG.clean_inference import load_controller
from app.ClickSEG.isegm.inference.clicker import Clicker
from config import DATA_DIR


def get_last_ckpt_path(ckpt_path):
    """Get the path of the last checkpoint in a directory"""
    assert isinstance(ckpt_path, Path)
    if ckpt_path.is_dir():
        ckpts = sorted(ckpt_path.glob("*.pth"))
        if len(ckpts):
            return ckpts[-1]
        else:
            return ckpt_path
    else:
        return sorted(ckpt_path.parent.glob("*.pth"))[-1]


def IoU(mask1, mask2):
    """Compute the intersection over union of two masks"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)


def mytorchload(ckpt_path):
    """Just like torch.load but checking that the file is ready"""
    lock_file_path = Path(str(ckpt_path) + ".lock")
    if os.path.exists(lock_file_path):
        time.sleep(0.5)  # this is more than enough
    with open(ckpt_path, "rb") as f:  # chatgpt recommends
        checkpoint = torch.load(f)
    return checkpoint


def get_updated_model(ckpt_path, model=None):
    """Reload the model if checkpoint has been updated"""
    assert isinstance(ckpt_path, Path)
    if ckpt_path.exists():
        last_ckpt_path = get_last_ckpt_path(ckpt_path)
        need_update = last_ckpt_path != ckpt_path or model is None
        if need_update and not last_ckpt_path.is_dir():
            return mytorchload(last_ckpt_path), last_ckpt_path
        return model, last_ckpt_path
    return model, ckpt_path


def annotate_img(img, preannotation, target, max_delay, threshold=0.9):
    external_clicker = Clicker(gt_mask=np.array(target))

    modelname = "focalclick"
    controller = load_controller()
    assert controller is not None

    # inference
    controller.set_image(np.array(img))
    pred_mask = np.zeros_like(np.array(target))
    if preannotation is not None:
        controller.set_mask(preannotation)
        pred_mask = preannotation

    number_of_clicks = 0
    if np.sum(target) == 0:
        if np.abs(np.sum(pred_mask)) > 0.001:
            time.sleep(
                max_delay * np.random.rand()
            )  # delay because we need to clear the bad prediction
        pred_mask = np.zeros_like(np.array(target))
    else:
        while (
            IoU(pred_mask, target) < threshold
            and number_of_clicks < 20
            and np.sum(target) != 0
        ):
            time.sleep(max_delay * np.random.rand())  # delay
            external_clicker.make_next_click(pred_mask)
            clicks_so_far = external_clicker.get_clicks()
            last_click = clicks_so_far[-1]
            (y, x) = last_click.coords
            is_positive = last_click.is_positive
            controller.add_click(x, y, is_positive)
            pred_mask = controller.result_mask
            number_of_clicks += 1
    time.sleep(max_delay * 0.1)  # time to say 'next'

    return pred_mask


def annotate_data(
    input_dir,
    target_dir,
    annotations_dir,
    ckpt_path=Path("checkpoints"),
    label=10,
    max_delay=4,
):
    Path(annotations_dir).mkdir(exist_ok=True, parents=True)
    totensor = torchvision.transforms.ToTensor()
    dataset = PerClassDataset(input_dir, target_dir, label)
    for i, (img, target) in enumerate(dataset):
        # check that the annotation does not already exist
        if (
            Path(annotations_dir)
            / f"annotation_{str(i).zfill(6)}_{str(label).zfill(2)}.png"
        ).exists():
            raise RuntimeError(f"Annotation {i} already exists")
        # update the model
        model, ckpt_path = get_updated_model(ckpt_path)
        # annotate the image
        preannotation = None
        if model is not None:
            device = next(model.decoder.parameters()).device
            preannotation = model(
                torchvision.transforms.functional.resize(
                    totensor(img).to(device), (480, 480), antialias=True
                ).unsqueeze(0)
            )
            # resize back to img size
            preannotation = (
                torchvision.transforms.functional.resize(
                    preannotation.squeeze(0), (img.height, img.width), antialias=True
                )
                .detach()
                .cpu()
                .numpy()[0]
            )
        annotation = annotate_img(img, preannotation, target, max_delay)
        # save the annotation
        Image.fromarray(annotation).save(
            Path(annotations_dir)
            / f"annotation_{str(i).zfill(6)}_{str(label).zfill(2)}.png"
        )
        print(
            "Annotated image",
            i,
            "with label",
            label,
            "and checkpoint",
            ckpt_path,
            "at time",
            datetime.datetime.utcnow(),
        )


if __name__ == "__main__":
    # get label as first argumetn of call
    label = int(sys.argv[1])
    input_dir = os.path.join(DATA_DIR, "perlabel_sbd/inputs")
    target_dir = os.path.join(DATA_DIR, "perlabel_sbd/targets")
    annotations_dir = os.path.join(DATA_DIR, "perlabel_sbd/annotations")
    annotate_data(input_dir, target_dir, annotations_dir, label=label, max_delay=4)
