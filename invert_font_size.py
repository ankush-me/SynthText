# Author: Ankush Gupta
# Date: 2015
"Script to generate font-models."

import pygame
from pygame import freetype
from text_utils import FontState
import numpy as np 
import matplotlib.pyplot as plt 
import _pickle as cp


pygame.init()


ys = np.arange(8,200)
A = np.c_[ys,np.ones_like(ys)]

xs = []
models = {} #linear model

FS = FontState()
#plt.figure()
for i in range(len(FS.fonts)):
	print(i)
	font = freetype.Font(FS.fonts[i], size=12)
	h = []
	for y in ys:
		h.append(font.get_sized_glyph_height(float(y)))
	h = np.array(h)
	m,_,_,_ = np.linalg.lstsq(A,h)
	models[font.name] = m
	print("{}:\t{}".format(i, font.name))
	xs.append(h)

with open('font_px2pt.cp','wb') as f:
	cp.dump(models,f)
#plt.plot(xs,ys[i])
#plt.show()
