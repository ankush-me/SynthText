# Author: Ankush Gupta
# Date: 2015

"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt 
import h5py
from PIL import Image

import configuration
from common import *
import cv2



def viz_textbb(text_im, charBB_list, wordBB, alpha=1.0 , image_name=None):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1)
    plt.imshow(text_im)
    #plt.savefig('results/{}.png'.format(image_name))
    H,W = text_im.shape[:2]

    # plot the character-BB:
    #for i in range(len(charBB_list)):
   #     bbs = charBB_list[i]
   #     ni = bbs.shape[-1]
   ##     for j in range(ni):
   #         bb = bbs[:,:,j]
    #        bb = np.c_[bb,bb[:,0]]
            #plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)

    # plot the word-BB:
    for i in range(wordBB.shape[-1]):
        bb = wordBB[:,:,i]
        bb = np.c_[bb,bb[:,0]]
        plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
        #visualize the indiv vertices:
        vcol = ['r','g','b','k']
        #for j in range(4):
        #    plt.scatter(bb[0,j],bb[1,j],color=vcol[j])

    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    #plt.show(block=False)
    plt.savefig('results/{}.jpg'.format(image_name))
    
def main(db_fname):
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print ("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))
    for k in dsets:
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']
        font = db['data'][k].attrs['font']
        j= k.replace(".", "_")
        viz_textbb(rgb, [charBB], wordBB, image_name=j )
        print ("image name        : ", colorize(Color.RED, k, bold=True))
        print ("  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1]))
        print ("  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1]))
        print ("  ** text         : ", colorize(Color.GREEN, txt))
        print ('  ** font         : ', colorize(Color.GREEN, font))
        #if 'q' in input("next? ('q' to exit) : "):
        #    break
    db.close()

if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='visualize Synthetic Scene-Text Images')
    
    parser.add_argument('--lang', default='ENG',
                        help='Select language : ENG/HI')
    args = parser.parse_args()


    configuration.lang = args.lang
    
    main('./SynthText_{}.h5'.format(configuration.lang))

