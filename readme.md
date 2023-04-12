# Dataset preparation 
- download NDD20 by running `bash segmentation_datasets/ndd20.sh` (change the path to your datadir in that file)


# To set up:
- Create `app/` dir
- `git clone https://github.com/franchesoni/clickseg_inference.git` inside `app/` dir
- rename the repo: `mv app/clickseg_inference app/ClickSEG`
- download `combined_segformerb3s2.pth` to `app/weights/focalclick_models`
- edit paths in `launch_both.sh` and `config.py`
- run `python -m segmentation_datasets.save_per_class` to create the SBD per-class dataset
