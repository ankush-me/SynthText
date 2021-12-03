
import h5py
import numpy as np
from PIL import Image
import os.path as osp
import _pickle as cp
import os
import numpy


def create_image_dataset(image_path, output_dir):
	myList = []
	img_db = {}
	if os.path.isdir(image_path):
		for i, file in enumerate(os.listdir(image_path)):
			myList.append(file)
			img = Image.open(osp.join(image_path, file)).convert('RGB')
			i = numpy.asarray(img)
			img_db[file] = numpy.asarray(img)
	
	with h5py.File("{}/img_db.h5".format(output_dir), "w") as f:
		for k, v in img_db.items():
			f.create_dataset(k, data=np.array(v, dtype=np.uint8))

import argparse

parser = argparse.ArgumentParser(description='Genereate image names file')
parser.add_argument('--image_path', default='./images',)
parser.add_argument('--output_dir', default='./', help="output base directory")

args = parser.parse_args()
create_image_dataset(args.image_path, args.output_dir)