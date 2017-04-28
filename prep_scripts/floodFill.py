"""
Python script to "flood-fill" the segments computed using gPb-UCM.
This assings the same integer label to all the pixels in the same segment.

Author: Ankush Gupta
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import h5py
import os.path as osp
import multiprocessing as mp
import traceback, sys

def get_seed(sx,sy,ucm):
    n = sx.size
    for i in xrange(n):
        if ucm[sx[i]+1,sy[i]+1] == 0:
            return (sy[i],sx[i])

def get_mask(ucm,viz=False):
    ucm = ucm.copy()
    h,w = ucm.shape[:2]
    mask = np.zeros((h-2,w-2),'float32')

    i = 0
    sx,sy = np.where(mask==0)
    seed = get_seed(sx,sy,ucm)
    areas = []
    labels=[]
    while seed is not None and i<1000:
        cv2.floodFill(mask,ucm,seed,i+1)
        # calculate the area (no. of pixels):
        areas.append(np.sum(mask==i+1))
        labels.append(i+1)

        # get the location of the next seed:
        sx,sy = np.where(mask==0)
        seed = get_seed(sx,sy,ucm)
        i += 1
    print "  > terminated in %d steps"%i

    if viz:
        plt.imshow(mask)
        plt.show()

    return mask,np.array(areas),np.array(labels)

def get_mask_parallel(ucm_imname):
    ucm,imname = ucm_imname
    try:
        return (get_mask(ucm.T),imname)
    except:
        return None
        #traceback.print_exc(file=sys.stdout)

def process_db_parallel(base_dir, th=0.11):
    """
    Get segmentation masks from gPb contours.
    """
    db_path = osp.join(base_dir,'ucm.mat')
    out_path = osp.join(base_dir,'seg_uint16.h5')
    # output h5 file:
    dbo = h5py.File(out_path,'w')
    dbo_mask = dbo.create_group("mask")

    class ucm_iterable(object):
        def __init__(self,ucm_path,th):
            self.th = th
            self.ucm_h5 = h5py.File(db_path,'r')
            self.N = self.ucm_h5['names'].size
            self.i = 0

        def __iter__(self):
            return self

        def get_imname(self,i):
            return "".join(map(chr, self.ucm_h5[self.ucm_h5['names'][0,self.i]][:]))

        def __stop__(self):
            print "DONE"
            self.ucm_h5.close()
            raise StopIteration

        def get_valid_name(self):
            if self.i >= self.N:
                self.__stop__()

            imname = self.get_imname(self.i)
            while self.i < self.N-1 and len(imname) < 4:
                self.i += 1
                imname = self.get_imname(self.i)

            if len(imname) < 4:
                self.__stop__()

            return imname

        def next(self):
            imname = self.get_valid_name()
            print "%d of %d"%(self.i+1,self.N)
            ucm = self.ucm_h5[self.ucm_h5['ucms'][0,self.i]][:]
            ucm = ucm.copy()
            self.i += 1
            return ((ucm>self.th).astype('uint8'),imname)

    ucm_iter = ucm_iterable(db_path,th)
    print "cpu count: ", mp.cpu_count()
    parpool = mp.Pool(4)
    ucm_result = parpool.imap_unordered(get_mask_parallel, ucm_iter, chunksize=1)

    for res in ucm_result:
        if res is None:
            continue
        ((mask,area,label),imname) = res
        print "got back : ", imname
        mask = mask.astype('uint16')
        mask_dset = dbo_mask.create_dataset(imname, data=mask)
        mask_dset.attrs['area'] = area
        mask_dset.attrs['label'] = label

    # close the h5 files:
    print "closing DB"
    dbo.close()
    print ">>>> DONE"


base_dir = '/home/' # directory containing the ucm.mat, i.e., output of run_ucm.m
process_db_parallel(base_dir)
