# Author: Ankush Gupta
# Date: 2015

"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division

import math
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
	
def viz_textbb(text_im, charBB_list, wordBB, alpha=1.0, image_name=None, text=None):
	"""
	text_im : image containing text
	charBB_list : list of 2x4xn_i bounding-box matrices
	wordBB : 2x4xm matrix of word coordinates
	"""
	
	for i in range(wordBB.shape[-1]):
		bb1 = wordBB[:, :, i]
		bb = convert_floating_coordinates_to_int(bb1).T
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
		cv2.imwrite('results/recog/{}_{}.png'.format(image_name,i), warped)

def main(db_fname):
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
		viz_textbb(rgb, [charBB], wordBB, image_name=j,text=txt)
		print("image name        : ", colorize(Color.RED, k, bold=True))
		print("  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1]))
		print("  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1]))
		print("  ** text         : ", colorize(Color.GREEN, txt))
		print('  ** font         : ', colorize(Color.GREEN, font))


if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description='crop images and create recogntion dataset')
	
	parser.add_argument('--lang', default='ENG',
	                    help='Select language : ENG/HI')
	args = parser.parse_args()
	
	configuration.lang = args.lang
	
	main('./SynthText_{}.h5'.format(configuration.lang))
