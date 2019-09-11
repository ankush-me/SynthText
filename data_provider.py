import os
from synthgen import *
from common import *
import wget
import tarfile


# TODO: move these contants inside DataProvider

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'  # TODO dedup
DB_FNAME = osp.join(DATA_PATH, 'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'


class DateProvider(object):

    def __init__(self, root_data_dir=None):
        # TODO: add option to override those 3:
        path_depth = "depth.h5"
        path_segmap = "seg.h5"
        self.path_images = "bg_img"
        self.db = None
        self.depth_db = None
        self.seg_db = None
        self.segmap = {}
        self.depth = {}

        if root_data_dir is None:
            # should download default example
            self.db = DateProvider.get_data()
            self.segmap = self.db['seg']
            self.depth = self.db['depth']
            self.imnames = sorted(self.db['image'].keys())
        else:
            # provided path to the folder with all data downloaded separately.
            # see https://github.com/ankush-me/SynthText#pre-processed-background-images
            self.path = root_data_dir
            self.depth_db = h5py.File(osp.join(self.path, path_depth), 'r')
            self.seg_db = h5py.File(osp.join(self.path, path_segmap), 'r')
            self.imnames = sorted(self.depth_db.keys())
            self.segmap = self.seg_db['mask']
            self.depth = self.depth_db

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

    def get_image(self, imname: str):
        if self.db is None:
            return Image.open(osp.join(self.path, self.path_images, imname)).convert('RGB')
        else:
            return Image.fromarray(self.db['image'][imname][:])

    def get_segmap(self, imname: str):
        return self.segmap[imname]

    def get_depth(self, imname: str):
        if self.db is None:
            return self.depth[imname][:].T[:, :, 0]
        else:
            return self.depth[imname][:].T[:, :, 1]

    def get_imnames(self):
        return self.imnames

    def close(self):
        if self.db is not None:
            self.db.close()
        if self.depth_db is not None:
            self.depth_db.close()
        if self.seg_db is not None:
            self.seg_db.close()
