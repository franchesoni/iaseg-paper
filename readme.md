# What this repo contains
## Introduction
The idea is to design a method to annotate a binary image segmentation dataset as fast as possible.
In order to do the simulations and evaluate the performance of such Intelligent Annotation (IA) schemes we need ground truth segmentations.
We use the "BELOW" set of images from the NDD20 dataset [1], which is a segmentation dataset with high mask quality as per SAM's user study [2] (only surpassed by iShape).
We decide not to work with iShape because the segmentation problem in iShape is i) instance segmentation and ii) harder. In contrast, the NDD20 dataset is real (iShape is synhtetic), easy (many times there's only one object per image) but out-of-domain (underwater images and dolphins). This allows us to evaluate annotation methods in the simplest setting.
Annotation of a binary segmentation dataset is, in practice, conducted by humans. Humans have access to tools such as brush and eraser, but the computer vision research community has come up with "smarter" ones. In particular, the area of interactive image segmentation studies how to quickly annotate a binary mask on an image considering user input. Reference works on this area include [2, 3, 4, 5]. Some of them also mix language, such as [2, 6], although text models of [2] weren't yet released and [6] is old. 

When annotating a binary segmentation dataset with interactive tools we have Intelligent Annotation (IA) on the image level, this is, an image is ideally annotated quickly, but we don't have it on the dataset level: the concepts captured by the annotations made so far aren't used at all to speed up further annotation. To tackle this problem we study the interaction between interactive image segmentation (IIS) models and a binary segmentation model M that can make predictions. If the predictions of M were perfect, the dataset can be automatically annotated. However, this is seldom the case (with [2] being kind of an exception). We are then facing the problem of how M (whatever model of the current concept) and an IIS model can cooperate to speed up annotation.

## Our work

### Dataset preparation
We use the underwater images of NDD20 and the corresponding masks. There are 2201 such images.
We divide the dataset into train (80%) and test (20%) where the train set corresponds to the first image names on the sorted list of image names. Although in our problem a test set isn't strictly needed, it's good to have it because it allows for comparison between segmentation networks M, e.g. against networks benchmarked on M.

### Supervised baseline
We train a model (torchvision's `deeplabv3_mobilenet_v3_large`) with weights trained on COCO, modifying the head to predict only one class. We use torchvision's `sigmoid_focal_loss` with default parameters. The optimizer is Adam with learning rate 0.001. We shuffle the dataset, use batch size 42, and drop the last batch if has another batch size. 


### Is a model helpful?
We annotate the data using diverse interactive image segmentation methods (SAM, FocalClick, RITM, SimpleClick) and using the same methods to correct a mask proposed by the trained image segmentation model. 


#### Results
[1] NDD20 dataset
[2] Segment Anything
[3] Reviving iterative training with mask guidance for interactive image segmentation
[4] FocalClick
[5] SimpleClick
[6] PhraseClick
[7] Language-driven semantic segmentation





# To-do
- reevaluate with actual image size

# Dataset preparation 
- download NDD20 by running `bash segmentation_datasets/ndd20.sh` (change the path to your datadir in that file)

# Experiments
- train `python -m supervised_baseline`  (comment out what's needed)
- evaluate `python -m supervised_baseline`  (comment out what's needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_loss         │   0.014407549984753132    │
│         test_mae          │     4.05710506439209      │
│         test_mse          │     17.89949607849121     │
└───────────────────────────┴───────────────────────────┘

# To set up:
- Create `app/` dir
- `git clone https://github.com/franchesoni/clickseg_inference.git` inside `app/` dir
- rename the repo: `mv app/clickseg_inference app/ClickSEG`
- download `combined_segformerb3s2.pth` to `app/weights/focalclick_models`
- edit paths in `launch_both.sh` and `config.py`
- run `python -m segmentation_datasets.save_per_class` to create the SBD per-class dataset
