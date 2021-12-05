# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os, sys, traceback
import os.path as osp

import configuration
import create_recognition_dataset
import visualize_results
from logger import wrap, entering, exiting
from synthgen import *
from common import *
import wget, tarfile

import lmdb

## Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 10  # no. of times to use the same image
SECS_PER_IMG = 5

# path to the data-file, containing image, depth and segmentation:

# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = './SynthText_{}.h5'.format(configuration.lang)
OUT_DIR = './'


def add_res_to_db(imgname, res, db):
    """
	Add the synthetically generated text image instance
	and other metadata to the dataset.
	"""
    ninstance = len(res)
    for i in range(ninstance):
        dname = "%s_%d" % (imgname, i)
        db['data'].create_dataset(dname, data=res[i]['img'])
        db['data'][dname].attrs['charBB'] = res[i]['charBB']
        db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
        db['data'][dname].attrs['font'] = res[i]['font']
        text_utf8 = [char.encode('utf8') for char in res[i]['txt']]
        db['data'][dname].attrs['txt'] = text_utf8


def save_res_to_imgs(imgname, res):
    """
	Add the synthetically generated text image instance
	and other metadata to the dataset.
	"""
    ninstance = len(res)
    for i in range(ninstance):
        filename = "{}/{}_{}.png".format(OUT_DIR, imgname, i)
        # Swap bgr to rgb so we can save into image file
        img = res[i]['img'][..., [2, 1, 0]]
        cv2.imwrite(filename, img)


@wrap(entering, exiting)
def main(data_path, viz=False):
    # open databases:
    print(colorize(Color.BLUE, 'getting data..', bold=True))

    # open the output h5 file:
    out_db = h5py.File(OUT_FILE, 'w')
    out_db.create_group('/data')
    print(colorize(Color.GREEN, 'Storing the output in: ' + OUT_FILE, bold=True))

    img_env = lmdb.open("img_lmdb", readonly=True)
    dep_env = lmdb.open("dep_lmdb", readonly=True)
    seg_env = lmdb.open("seg_lmdb", readonly=True)

    imnames = []

    with img_env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            imnames.add(key)

    print(imnames)

    # get the names of the image files in the dataset:
    imnames = sorted(imnames)

    N = len(imnames)
    global NUM_IMG
    if NUM_IMG < 0:
        NUM_IMG = N
    start_idx, end_idx = 0, min(NUM_IMG, N)

    RV3 = RendererV3(data_path, max_time=SECS_PER_IMG)  # TODO change max_time
    for i in range(start_idx, end_idx):

        imname = imnames[i]
        try:
            # get the image:
            # img = Image.fromarray(db['image'][imname][:])

            with img_env.begin() as txn:
                img = Image.fromarray(txn.get(imname))

            # get the pre-computed depth:
            #  there are 2 estimates of depth (represented as 2 "channels")
            #  here we are using the second one (in some cases it might be
            #  useful to use the other one):
            with dep_env.begin() as txn:
                depth = txn.get(imname)

            # depth = depth_db[imname][:]

            # depth = depth[:, :, 0]

            # get segmentation:
            # seg = seg_db["mask"][imname][:].astype('float32')
            # area = seg_db["mask"][imname].attrs['area']
            # label = seg_db["mask"][imname].attrs['label']

            with seg_env.begin() as txn:
                seg = txn.get(imname)
                area = txn.get(imname + "_areas")
                label = txn.get(imname + "_labels")


            # re-size uniformly:
            sz = depth.shape[:2][::-1]
            img = np.array(img.resize(sz, Image.ANTIALIAS))
            seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))
            # from utils import io
            # io.write_segm_img("seg.jpg", img, seg, alpha=0.5)
            # io.write_depth('depth', depth)
            # get_segmentation_crop(img, seg, label)

            print(colorize(Color.RED, '%d of %d' % (i, end_idx - 1), bold=True))
            res = RV3.render_text(img, depth, seg, area, label, ninstance=INSTANCE_PER_IMAGE, viz=viz)
            if len(res) > 0:
                # non-empty : successful in placing text:
                add_res_to_db(imname, res, out_db)
            # visualize the output:
            if viz:
                save_res_to_imgs(imname, res)
        # if 'q' in input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
        #  break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue

    out_db.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    parser.add_argument('--lang', default='ENG',
                        help='Select language : ENG/HI')

    parser.add_argument("--data_path", default="data/")
    parser.add_argument('--text_source', default='newsgroup/newsgroup.txt', help="text_source")
    args = parser.parse_args()

    configuration.lang = args.lang
    configuration.text_soruce = "newsgroup/newsgroup_{}.txt".format(args.lang)
    configuration.fontlist_file = "fonts/fontlist/fontlist_{}.txt".format(args.lang)
    configuration.char_freq_path = 'models/{}/char_freq.cp'.format(args.lang)
    configuration.font_px2pt = 'models/{}/font_px2pt.cp'.format(args.lang)
    OUT_FILE = './SynthText_{}.h5'.format(configuration.lang)

    main(args.data_path, args.viz)
# TODO remove this line. kept only for debuggging during development.
visualize_results.main('results/SynthText_{}.h5'.format(configuration.lang))
# create_recognition_dataset.main('results/SynthText_{}.h5'.format(configuration.lang))
