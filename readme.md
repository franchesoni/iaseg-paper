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
