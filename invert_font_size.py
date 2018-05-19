# Author: Ankush Gupta
# Date: 2015
"Script to generate font-models."

import pygame
from pygame import freetype
import os
import numpy as np 
import matplotlib.pyplot as plt 
import pickle as cp


pygame.init()


ys = np.arange(8,200)
A = np.c_[ys,np.ones_like(ys)]

xs = []
models = {} #linear model

FONT_LIST = './data/fonts/fontlist.txt'
fonts = [os.path.join('./data/fonts',f.strip()) for f in open(FONT_LIST)]
##plt.hold(True)
for i in range(len(fonts)):
	print(fonts[i])
	font = freetype.Font(fonts[i], size=12)
	h = []
	for y in ys:
		h.append(font.get_sized_glyph_height(int(y)))
	h = np.array(h)
	m,_,_,_ = np.linalg.lstsq(A,h)
	models[font.name] = m
	xs.append(h)

with open('./data/models/font_px2pt.cp','wb') as f:
	cp.dump(models,f)
#plt.plot(xs,ys[i])
#plt.show()
