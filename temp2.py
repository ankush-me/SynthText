import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.segmentation import felzenszwalb, flood_fill
from skimage.segmentation import mark_boundaries


img = io.imread('/home/shubham/Documents/MTP/datasets/imgs/COCO_train2014_000000001810.jpg')
segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=100)
from utils import io
print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')


io.write_segm_img("temp", img, segments_fz )
