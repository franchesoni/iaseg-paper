print("importing packages...")
import json
from PIL import Image
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry

from config import SAM_VITL_PATH, NDD20_DIR
from segmentation_datasets.ndd20 import NDD20Dataset, tonp_transform


norm_fn = lambda x: (x - x.min()) / (x.max() - x.min())

def show_mask(mask, ax, color=None):
    # from SAM
    if type(color) is str and color == "random":
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color is None:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def blend_masks(masks, colors):
    for ind in range(len(masks)):
        if ind == 0:
            mask_image = masks[ind] * colors[ind]
        else:
            mask_image += masks[ind] * colors[ind]
    return mask_image
    
    


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def dt(a):
    # from getting to 99 paper
    return cv2.distanceTransform((a * 255).astype(np.uint8), cv2.DIST_L2, 0)


def get_largest_incorrect_region(pred, gt):
    largest_incorrect_BF = []
    for val in [0, 1]:
        incorrect = (gt == val) * (pred != val)
        ret, labels_con = cv2.connectedComponents(incorrect.astype(np.uint8) * 255)
        label_unique, counts = np.unique(
            labels_con[labels_con != 0], return_counts=True
        )
        if len(counts) > 0:
            largest_incorrect = labels_con == label_unique[np.argmax(counts)]
            largest_incorrect_BF.append(largest_incorrect)
        else:
            largest_incorrect_BF.append(np.zeros_like(incorrect))

    largest_incorrect_cat = np.argmax(
        [np.count_nonzero(x) for x in largest_incorrect_BF]
    )
    largest_incorrect = largest_incorrect_BF[largest_incorrect_cat]
    return largest_incorrect, largest_incorrect_cat


def click_position_random(largest_incorrect):
    uys, uxs = np.where(largest_incorrect > 0)
    if uys.shape[0] == 0:
        return -1, -1
    i = np.random.randint(uys.shape[0])
    y, x = uys[i], uxs[i]
    return y, x


def click_position_center(largest_incorrect):
    # from getting to 99 paper
    h, w = largest_incorrect.shape

    largest_incorrect_boundary = np.zeros((h + 2, w + 2))
    largest_incorrect_boundary[1:-1, 1:-1] = largest_incorrect

    uys, uxs = np.where(largest_incorrect_boundary > 0)

    if uys.shape[0] == 0:
        return -1, -1

    no_click_mask = 1 - largest_incorrect_boundary
    dist = dt(1 - no_click_mask)
    dist = dist[1:-1, 1:-1]
    y, x = np.unravel_index(dist.argmax(), dist.shape)

    return y, x


def run_sam(category_id, visualize=False):
    torch.manual_seed(0)
    np.random.seed(0)

    print("loading dataset...")
    ndd20_train = NDD20Dataset(
        NDD20_DIR,
        subdataset="below",
        transforms=tonp_transform,
        split="train",
        reset=False,
    )
    ds = ndd20_train

    print("loading model...")
    sam = sam_model_registry["vit_l"](checkpoint=SAM_VITL_PATH)
    print("getting predictor...")
    predictor = SamPredictor(sam)

    performance = {}
    for sample_index, sample in enumerate(ds):
        img, mask, hw = sample["input"], sample["target"], sample["hw"]
        predictor.set_image(img)

        # start clicking
        performance[sample_index] = {"IoU": [], "rel_error": []}
        largest_incorrect = mask.copy()
        hat_mask = np.zeros_like(mask)
        largest_incorrect_cat = 1
        clicks = []
        click_labels = []
        prev_mask = None
        if visualize:
            plt.figure()
            plt.imshow(img)
            show_mask(mask, plt.gca(), color=np.array([1.0, 0.0, 0.0, 0.2]))
            plt.tight_layout()
            plt.savefig(f"visualization_gt.png")
        for n_click in range(20):
            # take the center click from inside mask using distance transform
            click_position_y, click_position_x = click_position_center(
                largest_incorrect
            )
            clicks.append([click_position_x, click_position_y])
            click_labels.append(largest_incorrect_cat)
            input_point = np.array(clicks)
            input_label = np.array(click_labels)

            hat_masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
                mask_input=prev_mask,
                return_logits=True,
            )

            logit_hat_mask = hat_masks[np.argmax(scores)]
            hat_mask = logit_hat_mask > predictor.model.mask_threshold
            prev_mask = logits[np.argmax(scores), :, :][None]

            # get the largest incorrect mask
            largest_incorrect, largest_incorrect_cat = get_largest_incorrect_region(
                hat_mask, mask
            )

            # compute intersection over union
            IoU = np.count_nonzero(hat_mask * mask) / np.count_nonzero(hat_mask + mask)
            performance[sample_index]["IoU"].append(IoU)  # index 0 is first click
            rel_error = largest_incorrect.sum() / mask.sum()
            performance[sample_index]["rel_error"].append(rel_error)
            if visualize:
                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                show_mask(hat_mask, plt.gca(), color=np.array([1.0, 0.0, 0.0, 0.5]))
                show_mask(mask, plt.gca(), color=np.array([0.0, 1.0, 1.0, 0.1]))
                show_points(input_point, input_label, plt.gca())
                plt.title(f"click {n_click}, IoU {IoU:.2f}")
                plt.axis("on")
                plt.tight_layout()
                plt.savefig(f"visualization.png")

                vis = (
                    norm_fn(logit_hat_mask)[..., None] * np.array([[[1.0, 0.0, 0.0]]]) 
                    + mask[..., None] * np.array([[[0.0, 1.0, 0.0]]]) * 0.75
                    + hat_mask[..., None] * np.array([[[0.0, 0.0, 1.0]]]) * 0.75
                )
                plt.imsave(f"visualization_error.png", vis)

                breakpoint()
            print("relative error", rel_error)
            if rel_error < 0.05:
                break
        print(
            f"finished sample {sample_index} with {n_click+1} clicks, IoU {IoU:.2f}, rel_error {rel_error:.2f}"
        )
    return performance


if __name__ == '__main__':
    performance = run_sam(37, visualize=False)
    # save it to json
    with open('sam_performance.json', 'w') as f:
        json.dump(performance, f)
