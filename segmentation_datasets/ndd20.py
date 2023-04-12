from torch.utils.data import Dataset
import shutil
import json
from pathlib import Path
import numpy as np
import torchvision
import cv2
from PIL import Image
import tqdm

def tonp_transform(img, mask):
    return np.array(img), np.array(mask)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((448, 448)),
    torchvision.transforms.ToTensor(),
])
def img_mask_transform(img, mask):
    img = transforms(img)
    mask = transforms(mask)
    return img, mask

class NDD20Dataset(Dataset):
    def __init__(self, root_dir, subdataset='below', transform=img_mask_transform, reset=False, split='train'):
        assert split in ['train', 'test'], "split must be 'train' or 'test'"
        assert subdataset.upper() in ['ABOVE', 'BELOW'], "subdataset must be 'above' or 'below'"
        self.subdataset = subdataset.upper()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.imgs_dir = self.root_dir / self.subdataset
        self.masks_dir = self.root_dir / f"{self.subdataset}_MASKS"
        self.split = split
        if reset:
            shutil.rmtree(self.masks_dir)
        if not self.masks_dir.exists():
            annotations_path = self.root_dir / f"{self.subdataset}_LABELS.json"
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
            full_annotation_keys = sorted(annotations.keys())
            self.generate_masks(annotations, full_annotation_keys)
        self.all_masks = sorted(self.masks_dir.glob("*.png"))
        self.all_imgs = sorted(self.imgs_dir.glob("*.jpg"))
        self.train_masks = self.all_masks[:int(len(self.all_masks)*0.8)]
        self.train_imgs = self.all_imgs[:int(len(self.all_imgs)*0.8)]
        self.test_masks = self.all_masks[int(len(self.all_masks)*0.8):]
        self.test_imgs = self.all_imgs[int(len(self.all_imgs)*0.8):]
        if split == 'train':
            self.imgs = self.train_imgs
            self.masks = self.train_masks
        elif split == 'test':
            self.imgs = self.test_imgs
            self.masks = self.test_masks
        self.check_default_format()



    def generate_masks(self, annotations, annotation_keys):
        self.masks_dir.mkdir(exist_ok=True)
        for ann_key in tqdm.tqdm(annotation_keys):
            img_name = annotations[ann_key]['filename']
            img = np.array(Image.open(self.imgs_dir / img_name))
            for region in annotations[ann_key]['regions']:
                polyline = region['shape_attributes']
                assert polyline['name'] == 'polyline', "Only polyline regions are supported"
                points = [(x, y) for x, y in zip(polyline['all_points_x'], polyline['all_points_y'])]
                mask = np.zeros_like(img)
                cv2.drawContours(mask, [np.array(points)], 0, (255, 255, 255), -1)
                mask = (0 < mask[..., 0])  # boolean mask
                mask = Image.fromarray(mask)  
                mask.save(self.masks_dir / f"mask_{img_name.split('.')[0]}.png")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        mask = Image.open(self.masks[idx])
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask

    def convert_to_mmlab(self):
        """Converts dataset to mmlab format"""
        self.check_default_format()
        # create train / val subfolders
        imgs_train_dir = self.imgs_dir / 'train'
        imgs_val_dir = self.imgs_dir / 'val'
        imgs_train_dir.mkdir(exist_ok=False)
        imgs_val_dir.mkdir(exist_ok=False)
        masks_train_dir = self.masks_dir / 'train'
        masks_val_dir = self.masks_dir / 'val'
        masks_train_dir.mkdir(exist_ok=False)
        masks_val_dir.mkdir(exist_ok=False)
        # move images and masks to train / val subfolders
        for img, mask in zip(self.train_imgs, self.train_masks):
            shutil.move(img, imgs_train_dir / img.name)
            shutil.move(mask, masks_train_dir / mask.name)
        for img, mask in zip(self.test_imgs, self.test_masks):
            shutil.move(img, imgs_val_dir / img.name)
            shutil.move(mask, masks_val_dir / mask.name)
        return {'seg_map_suffix': '_mask.png'}
            
    def check_default_format(self):
        # check that format is default, this is, two folders, one for images and one for masks
        assert self.masks_dir.exists(), "Masks directory does not exist"
        assert self.imgs_dir.exists(), "Images directory does not exist"
        # check that folders contain the images and masks
        assert len(list(self.masks_dir.glob("*.png"))) == len(list(self.imgs_dir.glob("*.jpg"))), "Number of masks and images is not the same"
        # check that masks are named mask_{img_name}.png
        assert all([mask.stem.split('_')[0] == 'mask' for mask in self.masks_dir.glob("*.png")]), "Masks are not named correctly"
        # check that images and masks are paired
        assert [mask.stem.split('_')[-1] for mask in self.all_masks] == [img.stem for img in self.all_imgs], "Masks and images are not aligned"
    
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
        shutil.rmtree(self.imgs_dir / 'train')
        shutil.rmtree(self.imgs_dir / 'val')
        shutil.rmtree(self.masks_dir / 'train')
        shutil.rmtree(self.masks_dir / 'val')
        # check everything is ok
        self.check_default_format()


def experimentation():
    import matplotlib.pyplot as plt
    ndd20dir = Path("/home/franchesoni/adisk/datasets/NDD20")
    jsonfiles = sorted(ndd20dir.glob("*.json"))
    with open(jsonfiles[0], 'r') as f: 
        above_labels = json.load(f)

    with open(jsonfiles[1], 'r') as f:
        below_labels = json.load(f)
    # *_labels is a dict. The keys are the image name and some strange id.
    # each value is a dict with keys "filename", "size", "file_attributes" and "regions"
    # when using "regions"
    # *_labels[key]["regions"] is a list (with 1 or 2 elements)
    # inside each element of a region we have shape attributes that determine a polyline

    exsample = below_labels[sorted(below_labels.keys())[0]]
    img = Image.open(ndd20dir / "BELOW" / exsample['filename'])
    img.save('img.png')
    polyline = exsample['regions'][0]['shape_attributes']
    plt.figure()
    plt.imshow(np.array(img))
    plt.plot(polyline['all_points_x'], polyline['all_points_y'], 'r')
    plt.savefig('border')

    import cv2
    points = [(x, y) for x, y in zip(polyline['all_points_x'], polyline['all_points_y'])]
    mask = np.zeros_like(img)[..., 0]  # 2D
    breakpoint()
    cv2.drawContours(mask, [np.array(points)], 0, (True), -1)
    Image.fromarray(mask).save('mask.png')


def visualize_first_images():
    dataset = NDD20Dataset(Path("/home/franchesoni/adisk/datasets/NDD20"), subdataset='below', reset=False)
    print(len(dataset))
    for ind, (img, mask) in enumerate(dataset):
        if not isinstance(img, np.ndarray):
            img, mask = tonp_transform(img, mask)
        print(ind, img.shape, mask.shape)
        Image.fromarray(img).save(f"img_{ind}.png")
        Image.fromarray(mask).save(f"mask_{ind}.png")
        if ind > 10:
            break   

    
if __name__ == '__main__':
    visualize_first_images()