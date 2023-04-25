from pathlib import Path
if config_name == "irrelevant":
    DATA_DIR = Path('/home/franchesoni/adisk/datasets')
elif config_name == "weird":
    DATA_DIR = Path("/home/franchesoni/adisk/datasets/iaseg-paper")
elif config_name == "jeanzay":
    DATA_DIR = Path('/gpfsscratch/rech/chl/uyz17rc/data/iaseg-paper')


DATA_DIR.mkdir(parents=True, exist_ok=True)
NDD20_DIR = DATA_DIR / "NDD20"
with open("segmentation_datasets/ndd20.sh", "r+") as f:
    contents = f.read()
    f.seek(0)
    f.write(f'DATASETS_DIR="{str(NDD20_DIR)}"\n' + contents)
    f.close()

DOWNLOAD_IF_NEEDED = True
SEED = 42
SAM_VITL_PATH = Path("weights/sam_vit_l_0b3195.pth")
if not SAM_VITL_PATH.exists():
    SAM_VITL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # run wget and wait until it finishes
    command = f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O {str(SAM_VITL_PATH)}"
    import os
    os.system(command)

# IMAGE_SIZE = (224, 224)
