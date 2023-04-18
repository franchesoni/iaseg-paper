from segmentation_datasets.sbd import get_sbd_dataset
import numpy as np

if __name__ == '__main__':
  sbd_dataset =  get_sbd_dataset()

  minh, maxh = 100000, 0
  for i, (img, target) in enumerate(sbd_dataset):
    # get height and width
    h, w, c = np.array(img).shape
    if h < minh:
      minh = h
    if h > maxh:  
      maxh = h
    print(i, minh, maxh)
  print(minh, maxh)
