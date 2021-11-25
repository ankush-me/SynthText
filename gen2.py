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
import visualize_results
from logger import wrap, entering, exiting
from synthgen import *
from common import *
import wget, tarfile

## Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 10  # no. of times to use the same image
SECS_PER_IMG = 5

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH, 'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText_{}.h5'.format(configuration.lang)
OUT_DIR = 'results'


def get_data():
	"""
	Download the image,depth and segmentation data:
	Returns, the h5 database.
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
def main(viz=False):
	# open databases:
	print(colorize(Color.BLUE, 'getting data..', bold=True))
	db = get_data()
	print(colorize(Color.BLUE, '\t-> done', bold=True))
	
	# open the output h5 file:
	out_db = h5py.File(OUT_FILE, 'w')
	out_db.create_group('/data')
	print(colorize(Color.GREEN, 'Storing the output in: ' + OUT_FILE, bold=True))
	
	img_db = h5py.File("h5py_files/img_db.h5", "r")
	depth_db = h5py.File('h5py_files/depth.h5', 'r')
	seg_db = h5py.File('h5py_files/seg_uint16.h5', 'r')
	
	# get the names of the image files in the dataset:
	imnames = sorted(img_db.keys())
	N = len(imnames)
	global NUM_IMG
	if NUM_IMG < 0:
		NUM_IMG = N
	start_idx, end_idx = 0, min(NUM_IMG, N)
	
	RV3 = RendererV3(DATA_PATH, max_time=SECS_PER_IMG)  # TODO change max_time
	for i in range(start_idx, end_idx):
		imname = imnames[i]
		try:
			# get the image:
			# img = Image.fromarray(db['image'][imname][:])
			
			img = Image.fromarray(img_db[imname][:])
			# get the pre-computed depth:
			#  there are 2 estimates of depth (represented as 2 "channels")
			#  here we are using the second one (in some cases it might be
			#  useful to use the other one):
			depth = depth_db[imname][:].T
			depth = depth[:, :, 1]
			# get segmentation:
			
			seg = seg_db["mask"][imname][:].astype('float32')
			area = seg_db["mask"][imname].attrs['area']
			label = seg_db["mask"][imname].attrs['label']
			
			# re-size uniformly:
			sz = depth.shape[:2][::-1]
			img = np.array(img.resize(sz, Image.ANTIALIAS))
			seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))
			#get_segmentation_crop(img, seg, label)
			
			print(colorize(Color.RED, '%d of %d' % (i, end_idx - 1), bold=True))
			res = RV3.render_text(img, depth, seg, area, label,
			                      ninstance=INSTANCE_PER_IMAGE, viz=viz)
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
	db.close()
	out_db.close()


if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
	parser.add_argument('--viz', action='store_true', dest='viz', default=False,
	                    help='flag for turning on visualizations')
	parser.add_argument('--lang', default='ENG',
	                    help='Select language : ENG/HI')
	
	parser.add_argument('--text_source', default='newsgroup/newsgroup.txt', help="text_source")
	args = parser.parse_args()
	
	configuration.lang = args.lang
	configuration.text_soruce = "newsgroup/newsgroup_{}.txt".format(args.lang)
	configuration.fontlist_file = "fonts/fontlist/fontlist_{}.txt".format(args.lang)
	configuration.char_freq_path = 'models/{}/char_freq.cp'.format(args.lang)
	configuration.font_px2pt = 'models/{}/font_px2pt.cp'.format(args.lang)
	OUT_FILE = 'results/SynthText_{}.h5'.format(configuration.lang)
	
	main(args.viz)
	# TODO remove this line. kept only for debuggging during development.
	visualize_results.main('results/SynthText_{}.h5'.format(configuration.lang))
