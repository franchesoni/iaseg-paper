from pathlib import Path
DATA_DIR = Path('/home/franchesoni/adisk/datasets')
DATA_DIR.mkdir(parents=True, exist_ok=True)
NDD20_DIR = DATA_DIR / "NDD20"
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
