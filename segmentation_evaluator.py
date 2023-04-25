from torch import log, nanmean
import json
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
import numpy as np
from torchvision.ops import sigmoid_focal_loss
from torch.nn import MSELoss, L1Loss
from pathlib import Path

from segmentation_datasets.ndd20 import NDD20Dataset
from config import DATA_DIR

def mseloss(y_hat, y):
    return MSELoss()(y_hat, y.float())

def l1loss(y_hat, y):
    return L1Loss()(y_hat, y.float())

def focal_loss(y_hat, y, apply_sigmoid=False):
    y = y.float()
    if apply_sigmoid:
        return sigmoid_focal_loss(y_hat, y, reduction="mean")
    else:
        invsigmoid_y_hat = log(y_hat / (1 - y_hat))
        # apply inverse sigmoid
        return nanmean(sigmoid_focal_loss(invsigmoid_y_hat, y, reduction="none"))

def get_binary_pred(y_hat, threshold=0.5):
    assert y_hat.max() <= 1 and y_hat.min() >= 0
    return y_hat > threshold

def get_tp(binary_pred, y):
    tp = (binary_pred & y).sum()
    return tp

def get_fp(binary_pred, y):
    fp = (binary_pred & ~y).sum()
    return fp

def get_tn(binary_pred, y):
    tn = (~binary_pred & ~y).sum()
    return tn

def get_fn(binary_pred, y):
    fn = (~binary_pred & y).sum()
    return fn

def get_iou(tp, fp, fn):
    # also called jaccard index
    return tp / (tp + fp + fn)

def get_dice(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)

def get_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def get_precision(tp, fp):
    if (tp+fp) == 0:
        return tp
    return tp / (tp + fp)

def get_recall(tp, fn):
    return tp / (tp + fn)

def get_specificity(tn, fp):
    return tn / (tn + fp)

def get_results_for_thresholds(y_hat, y, n_thresholds=10):
    if y_hat.max() > 1:
        # assume it's in [0, 255]
        y_hat = y_hat / 255
    results = {}
    for threshold in np.linspace(0, 1, n_thresholds):
        bpred = get_binary_pred(y_hat, threshold)
        tp, fp, tn, fn = get_tp(bpred, y), get_fp(bpred, y), get_tn(bpred, y), get_fn(bpred, y)
        iou, dice, acc, prec, rec, spec = get_iou(tp, fp, fn), get_dice(tp, fp, fn), get_accuracy(tp, tn, fp, fn), get_precision(tp, fp), get_recall(tp, fn), get_specificity(tn, fp)
        results[threshold] = {'iou': iou, 'dice': dice, 'acc': acc, 'prec': prec, 'rec': rec, 'spec': spec, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    return results

def get_average_results(y_hat, y, n_thresholds=10):
    y = y > 0  # convert to bool if needed
    results = get_results_for_thresholds(y_hat, y, n_thresholds)
    return {k: np.mean([v[k] for v in results.values()]) for k in results[0].keys()}



reg_metrics = {
    'mse': mseloss,
    'mae': l1loss,
    'focal_loss': focal_loss,
}
cls_metrics = {
    'classification_metrics': get_average_results,
}
metrics = {**reg_metrics, **cls_metrics}

def evaluate_predictions(pred_paths, gt_paths):
    assert len(pred_paths) == len(gt_paths)
    results = {}
    for ind, (pred_path, gt_path) in enumerate(zip(pred_paths, gt_paths)):
        pred = Image.open(pred_path)
        gt = Image.open(gt_path)
        pred = np.array(pred)
        gt = np.array(gt)
        for metric_name, metric in metrics.items():
            if metric_name not in results:
                results[metric_name] = []
            if callable(metric):
                results[metric_name].append(metric(pred, gt))
            else:
                raise NotImplementedError(f"Metric {metric_name} is not implemented.")

def evaluate_ndd20_predictions(pred_dir='predictions', per_threshold=False):
    totensor = ToTensor()
    pred_paths = sorted(Path(pred_dir).glob("*.png"))
    pred_indices = [int(p.stem) for p in pred_paths]
    ndd20 = NDD20Dataset(
        Path(DATA_DIR) / "NDD20",
        subdataset="below",
        transforms=None,
        split="all",
        reset=False,
    )
    gt_paths = [maskpath for maskpath in ndd20.all_masks if int(maskpath.stem.split('_')[-1]) in pred_indices]
    assert len(pred_paths) == len(gt_paths)
    results = []
    for ind, (pred_path, gt_path) in tqdm.tqdm(enumerate(zip(pred_paths, gt_paths))):
        pred = Image.open(pred_path).convert('L')
        gt = Image.open(gt_path)
        pred = totensor(np.array(pred) / 255)
        gt = totensor(np.array(gt))
        if per_threshold:
            results.append(get_results_for_thresholds(pred, gt))
        else:
            results.append({})
            for metric_name, metric in metrics.items():
                results[-1][metric_name] = metric(pred, gt)
        # if ind > 10:
        #     break
    if per_threshold:
        return results
    # put classification metrics along regression losses
    new_results = []
    for ind, result in enumerate(results):
        cls_metrics = result.pop('classification_metrics')
        result = result | cls_metrics
        new_results.append(result)
    return new_results 

def visualize_results_barplot(results):
    plt.figure()
    scale = {key: np.max(np.abs([result[key] for result in results])) for key in results[0].keys()}
    for ind, result in enumerate(results):
        # add all metrics for one of the results to bar plot
        for key_ind, (key, value) in enumerate(result.items()):
            plt.bar(ind+key_ind / (len(result)+1), value / scale[key], 1/ (len(result)+1), label=key)
            plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results.png')

def visualize_results_cross_correlation(results):
    plt.figure()
    # convert results to array
    keys = results[0].keys()
    results = np.array([[result[key] for key in keys] for result in results])  # R x K
    # compute correlation matrix
    corr = np.corrcoef(results, rowvar=False)
    # plot corr 
    plt.imshow(corr, cmap='hot', interpolation='nearest')
    plt.xticks(range(len(keys)), keys, rotation=90)
    plt.yticks(range(len(keys)), keys)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('results_corr.png')

def visualize_results_precrecspec(results):

    # convert results to array
    thresholds = list(results[0].keys())
    metrics = list(results[0][thresholds[0]].keys())
    # average over all images
    results = np.array([[[result[threshold][metric] for threshold in thresholds] for metric in metrics] for result in results])  # R x M x T
    results = np.mean(results, axis=0)  # M x T
    precrecspec_indices = metrics.index('prec'), metrics.index('rec'), metrics.index('spec')
    results = results[precrecspec_indices, :]  # 3 x T
    best_threshold_ind = np.argmax(results[0, :] * results[1, :] * results[2, :])
    print(f"Best threshold: {thresholds[best_threshold_ind]} of ind {best_threshold_ind}")
    # computes the mean 3d surface of precision, recall, specificity
    # each point of the surface corresponds to a threshold
    # the surface is then averaged over all results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot3D(results[0, :], results[1, :], results[2, :], c='r', marker='o')
    ax.scatter(np.array([0, 0, 0, 0, 1, 1, 1, 1]), np.array([0, 0, 1, 1, 0, 0, 1, 1]), np.array([0, 1, 0, 1, 0, 1, 0, 1]), c='b', marker='x')
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_zlabel('Specificity')
    plt.tight_layout()
    ax.view_init(30, 30)
    plt.savefig('results_precrecspec_0.png')
    ax.view_init(30, 60)
    plt.savefig('results_precrecspec_1.png')
    ax.view_init(60, 30)
    plt.savefig('results_precrecspec_2.png')
    return thresholds[best_threshold_ind]

def report_ndd20():
    results = evaluate_ndd20_predictions()
    visualize_results_cross_correlation(results)
    perthreshresults = evaluate_ndd20_predictions(per_threshold=True)
    best_thresh = visualize_results_precrecspec(perthreshresults)
    print(f"Best threshold: {best_thresh}")
    mean_mae = np.mean([result['mae'] for result in results])
    print(f"Mean MAE: {mean_mae:.3f}")
    mean_best_iou = np.mean([perthreshresult[best_thresh]['iou'] for perthreshresult in perthreshresults])
    print(f"Mean IoU: {mean_best_iou:.3f}")
    with open("results_ndd20.json", "w") as f:
        json.dump({
            "best_threshold": best_thresh,
            "mean_mae": mean_mae,
            "mean_best_iou": mean_best_iou,
            }, f, indent=4)


if __name__ == "__main__":
    report_ndd20()
