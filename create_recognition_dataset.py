# Author: Ankush Gupta
# Date: 2015

"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division

import math
import os.path

import h5py


import configuration
from common import *
from math import floor, ceil

import numpy as np
import cv2

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	
	rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
	tr, br = rightMost
	
	x1, y1 = bl
	x2, y2 = br
	w1 =  math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
	
	x1, y1 = tl
	x2, y2 = tr
	w2 =  math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
	#width of box
	w = max(w1, w2)
	
	x1, y1 = tl
	x2, y2 = bl
	h1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
	
	x1, y1 = tr
	x2, y2 = br
	h2 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
	h=max(h1, h2)  # height of box
	
	return np.array([tl, tr, br, bl], dtype="float32"), int(w) , int(h)
def convert_floating_coordinates_to_int(bb):

	bb[0][0] = floor(bb[0][0])
	bb[0][1] = ceil(bb[0][1])
	bb[0][2] = ceil(bb[0][2])
	bb[0][3] = floor(bb[0][3])
	bb[1][0] = floor(bb[1][0])
	bb[1][1] = floor(bb[1][1])
	bb[1][2] = ceil(bb[1][2])
	bb[1][3] = ceil(bb[1][3])
	bb = np.where(bb < 0, 0, bb)
	return bb.astype(int)

def get_string_representation_of_bbox(bb):
	return "{},{},{},{},{},{},{},{}".format(bb[0][0], bb[0][1],bb[1][0], bb[1][1],bb[2][0], bb[2][1],bb[3][0], bb[3][1])
def crop_words(text_im, wordBB, image_name=None, text=None):
	"""
	text_im : image containing text
	charBB_list : list of 2x4xn_i bounding-box matrices
	wordBB : 2x4xm matrix of word coordinates
	"""
	

	bb = convert_floating_coordinates_to_int(wordBB).T
	bb,width , height = order_points(bb)
	src_pts = bb.astype("float32")
	dst_pts = np.array([
	                    [0, 0],
	                    [width - 1, 0],
	                    [width - 1, height - 1],
	[0, height - 1]], dtype="float32")
	
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)

	warped = cv2.warpPerspective(text_im, M, (width, height))
	warped  = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
	
	
	min_x = min(bb[0][0],bb[1][0],bb[2][0],bb[3][0])
	max_x = max(bb[0][0], bb[1][0], bb[2][0], bb[3][0])
	
	min_y = min(bb[0][1],bb[1][1],bb[2][1],bb[3][1])
	max_y = max(bb[0][1], bb[1][1], bb[2][1], bb[3][1])
	
	unwarped = text_im[min_x : max_x, min_y:max_y]
	
	
	# bouding boxes are cropped using 2 techqniques, warped and unwarped both represent different challenges for text
	# recognition model.
	
	return warped , unwarped


def create_recognition_dataset_from_db_file(db_fname):
	db = h5py.File(db_fname, 'r')
	dsets = sorted(db['data'].keys())
	print("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))
	for k in dsets:
		rgb = db['data'][k][...]
		charBB = db['data'][k].attrs['charBB']
		wordBB = db['data'][k].attrs['wordBB']
		txt = db['data'][k].attrs['txt']
		font = db['data'][k].attrs['font']
		
		j = k.replace(".", "_")
		crop_words(rgb,wordBB, image_name=j, text=txt)
		print("image name        : ", colorize(Color.RED, k, bold=True))
		print("  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1]))
		print("  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1]))
		print("  ** text         : ", colorize(Color.GREEN, txt))
		print('  ** font         : ', colorize(Color.GREEN, font))
		
		
def create_recognition_dataset_from_detection_datset_folder(path, gt_file):
	lines = gt_file.readlines()
	for line in lines:
		img_path, word_bb, text, font = line.split("\t")
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		word_bb= word_bb.split(",")
		word_bb = np.array([int(bb) for bb in word_bb]).reshape((4,2))
		img_name = os.path.basename(img_path)
		crop_words(img, word_bb, image_name=img_name, text=text)
		print("image name        : ", colorize(Color.RED, bold=True))
		print("  ** text         : ", colorize(Color.GREEN, text))
		print('  ** font         : ', colorize(Color.GREEN, font))
		
def main(db_fname):
	create_recognition_dataset_from_detection_datset_folder()


if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description='crop images and create recogntion dataset')
	
	parser.add_argument('--lang', default='ENG',
	                    help='Select language : ENG/HI')
	
	parser.add_argument('--path', default='./output',
	                    help='Select language : ENG/HI')
	
	parser.add_argument('--gt_file', default='./output/gt.txt',
	                    help='Select language : ENG/HI')
	
	args = parser.parse_args()
	
	configuration.lang = args.lang
	
	main('./SynthText_{}.h5'.format(configuration.lang))
