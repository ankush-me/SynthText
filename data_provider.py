import h5py
import numpy as np
from PIL import Image
import os.path as osp
# import cPickle as cp
import imageio
import numpy as np
import h5py
import os
import sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget
import tarfile
from functools import reduce
import re
from time import time


# TODO: move these contants inside DataProvider

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH, 'dset.h5')

# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText.h5'


class DateProvider(object):

    @staticmethod
    def get_data():
        """
        Downloads the archive using link specified in DATA_URL. Unpacks the archive, treats it as h5 database.
        The image, depth and segmentation data is downloaded.

        Returns:
            the h5 database.
        """
        if not osp.exists(DB_FNAME):
            try:
                colorprint(Color.BLUE, '\tdownloading data (56 M) from: ' + DATA_URL, bold=True)
                print()
                sys.stdout.flush()
                out_fname = 'data.tar.gz'
                wget.download(DATA_URL, out=out_fname)
                tar = tarfile.open(out_fname)
                tar.extractall()
                tar.close()
                os.remove(out_fname)
                colorprint(Color.BLUE, '\n\tdata saved at:' + DB_FNAME, bold=True)
                sys.stdout.flush()
            except:
                print(colorize(Color.RED, 'Data not found and have problems downloading.', bold=True))
                sys.stdout.flush()
                sys.exit(-1)
        # open the h5 file and return:
        return h5py.File(DB_FNAME, 'r')

    #
    # def get_data(self, path: str):
    #     # path_names = "imnames.cp"
    #     path_images = "bg_img"
    #     path_depth = "depth.h5"
    #     path_segmap = "seg.h5"
    #
    #     depth_db = h5py.File(path + "/" + path_depth, 'r')
    #     seg_db = h5py.File(path + "/" + path_segmap, 'r')
    #     imnames = sorted(depth_db.keys())
    #
    #     img = Image.open(osp.join(path, path_images, imname)).convert('RGB')
    #
    #     # img = Image.fromarray(db['image'][imname][:])
    #
    # @staticmethod
    # def say_hello():
    #    print("hello")
    #
    #
    #