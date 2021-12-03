# Author: Ankush Gupta
# Date: 2015
"Script to generate font-models."

import pygame
from pygame import freetype

import configuration
from text_utils import FontState
import numpy as np 
import matplotlib.pyplot as plt 
import _pickle as cp


pygame.init()


def invert_font_size(data_path):
	ys = np.arange(8, 200)
	A = np.c_[ys, np.ones_like(ys)]
	xs = []
	models = {}  # linear model
	FS = FontState()
	# plt.figure()
	for i in range(len(FS.fonts)):
		print(i)
		font = freetype.Font(FS.fonts[i], size=12)
		h = []
		for y in ys:
			h.append(font.get_sized_glyph_height(float(y)))
		h = np.array(h)
		m, _, _, _ = np.linalg.lstsq(A, h)
		models[font.name] = m
		print("{}:\t{}".format(i, font.name))
		xs.append(h)
	with open("{}/{}".format(data_path,configuration.font_px2pt), 'wb') as f:
		cp.dump(models, f)
	
import argparse

parser = argparse.ArgumentParser(description='invert font size')
parser.add_argument('--lang', default='ENG',
                    help='Select language : ENG/HI')
parser.add_argument("--data_path", default="data/")
args = parser.parse_args()
configuration.char_freq_path = 'models/{}/char_freq.cp'.format(args.lang)
configuration.font_px2pt = 'models/{}/font_px2pt.cp'.format(args.lang)

invert_font_size(args.data_path)
