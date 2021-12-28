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
import os
import random

import tarfile
import wget

from common import *
from create_recognition_dataset import convert_floating_coordinates_to_int, order_points, \
	get_string_representation_of_bbox
from synthgen import *

## Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 10  # no. of times to use the same image
SECS_PER_IMG = 5

# path to the data-file, containing image, depth and segmentation:

# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = './SynthText_{}.h5'.format(configuration.lang)


def get_data(data_path):
	"""
	Download the image,depth and segmentation data:
	Returns, the h5 database.
	"""
	DB_FNAME = osp.join(data_path, 'dset.h5')
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
		db['data'][dname].attrs['font'] = res[i]['font']
		text_utf8 = [char.encode('utf8') for char in res[i]['txt']]
		db['data'][dname].attrs['txt'] = text_utf8


def save_res_to_imgs(imgname, res, gt_file, out_dir):
	"""
	Add the synthetically generated text image instance
	and other metadata to the dataset.
	"""
	ninstance = len(res)
	for i in range(ninstance):
		filename = "{}/{}_{}.jpg".format(os.path.basename(out_dir), imgname, i)
		img_file_name = "{}/{}_{}.jpg".format(out_dir, imgname, i)
		for j in range(len(res[i]["txt"])):
			bb = convert_floating_coordinates_to_int(res[i]['wordBB'][:,:,j]).T
			bb,_,_ = order_points(bb)
			bb = get_string_representation_of_bbox(bb)
			s = "{}\t{}\t{}\t{}".format(filename, bb,res[i]['txt'][j] ,res[i]['font'][j])
			gt_file.write(s)
			gt_file.write("\n")
		# Swap bgr to rgb so we can save into image file
		img = res[i]['img'][..., [2, 1, 0]]
		cv2.imwrite(img_file_name, img)
		

@wrap(entering, exiting)
def main(data_path,depth_dir, img_dir, gt_file_name,out_dir,  viz=False):
	# open databases:
	print(colorize(Color.BLUE, 'getting data..', bold=True))
	print(colorize(Color.BLUE, '\t-> done', bold=True))
	
	# open the output h5 file:
	#out_db = h5py.File(OUT_FILE, 'w')
	#out_db.create_group('/data')
	#print(colorize(Color.GREEN, 'Storing the output in: ' + OUT_FILE, bold=True))
	
	#img_db = h5py.File("./img_db.h5", "r")
	depth_db = h5py.File("{}/depth.h5".format(depth_dir), 'r')
	seg_db = h5py.File("{}/seg.h5".format(depth_dir), 'r')
	
	# get the names of the image files in the dataset:
	imnames = sorted(open("{}/image_names.txt".format(depth_dir)).readlines())
	#imnames = sorted(img_db.keys())
	N = len(imnames)
	global NUM_IMG
	if NUM_IMG < 0:
		NUM_IMG = N
	start_idx, end_idx = 0, min(NUM_IMG, N)
	
	RV3 = RendererV3(data_path, max_time=SECS_PER_IMG)
	gt_file = open("{}/{}".format(out_dir,gt_file_name), "w")
	range_list= list(range(start_idx, end_idx))
	random.shuffle(range_list)
	
	for i in range(start_idx, end_idx):

		imname = imnames[range_list[i]].strip()
		try:
			# get the image:
			# img = Image.fromarray(db['image'][imname][:])
			
			#img = Image.fromarray(img_db[imname][:])
			
			# get the pre-computed depth:
			#  there are 2 estimates of depth (represented as 2 "channels")
			#  here we are using the second one (in some cases it might be
			#  useful to use the other one):
			if imname not in depth_db:
				continue
				
			img = cv2.imread("{}/{}".format(img_dir , imname))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			
			depth = depth_db[imname][:]
			
			# depth = depth[:, :, 0]
			
			# get segmentation:
			seg = seg_db["mask"][imname][:].astype('float32')
			area = seg_db["mask"][imname].attrs['area']
			label = seg_db["mask"][imname].attrs['label']
			
			# re-size uniformly:
			#sz = depth.shape[:2][::-1]
			#img = np.array(img.resize(sz, Image.ANTIALIAS))
			#seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))
			from utils import io
			#io.write_segm_img("seg.jpg", img, seg, alpha=0.5)
			#io.write_depth('depth', depth)
			# get_segmentation_crop(img, seg, label)
			
			print(colorize(Color.RED, '%d of %d' % (i, end_idx - 1), bold=True))
			res = RV3.render_text(img, depth, seg, area, label,
			                      ninstance=INSTANCE_PER_IMAGE, viz=viz)
			if len(res) > 0:
				# non-empty : successful in placing text:
				# add_res_to_db(imname, res, out_db)
				save_res_to_imgs("{}_{}".format(gt_file_name[0:gt_file_name.find(".")],i), res, gt_file, out_dir)
				gt_file.flush()
				
			# visualize the output:
			if viz:
				save_res_to_imgs(imname, res, gt_file, out_dir)
		# if 'q' in input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
		#  break
		except:
			traceback.print_exc()
			print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
			continue
	#out_db.close()
	gt_file.close()

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
	parser.add_argument('--viz', action='store_true', dest='viz', default=False,
	                    help='flag for turning on visualizations')
	parser.add_argument('--lang',
	                    help='Select language : ENG/HI')
	
	parser.add_argument("--data_path", default="data/")
	parser.add_argument('--text_source', default='newsgroup/newsgroup.txt', help="text_source")
	parser.add_argument("--image_dir", default="./", help="path to images")
	parser.add_argument("--depth_dir", default="./", help="path to depth map and seg map")
	parser.add_argument("--gt_file", default="gt.txt", help="path to output gt file")
	parser.add_argument("--out_dir", default="./output", help="path to output gt file")
	args = parser.parse_args()
	
	configuration.lang = args.lang
	configuration.text_soruce = "newsgroup/newsgroup_{}.txt".format(args.lang)
	configuration.fontlist_file = "fonts/fontlist/fontlist_{}.txt".format(args.lang)
	configuration.char_freq_path = 'models/{}/char_freq.cp'.format(args.lang)
	configuration.font_px2pt = 'models/{}/font_px2pt.cp'.format(args.lang)
	OUT_FILE = './SynthText_{}.h5'.format(configuration.lang)
	
	main(args.data_path,args.depth_dir,args.image_dir, args.gt_file,args.out_dir, args.viz)
# TODO remove this line. kept only for debugging during development.
# visualize_results.main('results/SynthText_{}.h5'.format(configuration.lang))
# create_recognition_dataset.main('results/SynthText_{}.h5'.format(configuration.lang))
