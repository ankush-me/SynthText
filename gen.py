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
from synthgen import *
from common import *
from functools import reduce
import re
from time import time
from data_provider import DateProvider


# Define some configuration variables:
NUM_IMG = 1  # number of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1  # number of times to use the same image
SECS_PER_IMG = 5  # max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
OUT_FILE = 'results/SynthText.h5'

MASKS_DIR = "./masks"


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
        L = res[i]['txt']
        L = [n.encode("ascii", "ignore") for n in L]
        db['data'][dname].attrs['txt'] = L


def main(viz=False, debug=False, output_masks=False, data_path=None):
    """
    Entry point.

    Args:
        viz: display generated images. If this flag is true, needs user input to continue with every loop iteration.
        output_masks: output masks of text, which was used during generation
    """
    if output_masks:
        # create a directory if not exists for masks
        if not os.path.exists(MASKS_DIR):
            os.makedirs(MASKS_DIR)

    # open databases:
    print(colorize(Color.BLUE, 'getting data..', bold=True))

    provider = DateProvider(data_path)

    # db = DateProvider.get_data()
    print(colorize(Color.BLUE, '\t-> done', bold=True))

    # open the output h5 file:
    out_db = h5py.File(OUT_FILE, 'w')
    out_db.create_group('/data')
    print(colorize(Color.GREEN, 'Storing the output in: ' + OUT_FILE, bold=True))

    # get the names of the image files in the dataset:
    imnames = provider.get_imnames()
    N = len(imnames)
    global NUM_IMG
    if NUM_IMG < 0:
        NUM_IMG = N
    start_idx, end_idx = 0, min(NUM_IMG, N)

    renderer = RendererV3(DATA_PATH, max_time=SECS_PER_IMG)
    for i in range(start_idx, end_idx):
        imname = imnames[i]

        try:
            # get the image:
            img = provider.get_image(imname)
            # get the pre-computed depth:
            #  there are 2 estimates of depth (represented as 2 "channels")
            #  here we are using the second one (in some cases it might be
            #  useful to use the other one):
            depth = provider.get_depth(imname)
            # get segmentation:
            seg = provider.get_segmap(imname)[:].astype('float32')
            area = provider.get_segmap(imname).attrs['area']  # number of pixels in each region
            label = provider.get_segmap(imname).attrs['label']

            # re-size uniformly:
            sz = depth.shape[:2][::-1]
            img = np.array(img.resize(sz, Image.ANTIALIAS))
            seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))
            print(colorize(Color.RED, '%d of %d' % (i, end_idx - 1), bold=True))

            if debug:
                print("\n    Processing " + str(imname) + "...")

            res = renderer.render_text(img, depth, seg, area, label,
                                  ninstance=INSTANCE_PER_IMAGE)
            if len(res) > 0:
                # non-empty : successful in placing text:
                add_res_to_db(imname, res, out_db)
                if debug:
                    print("    Success. " + str(len(res[0]['txt'])) + " texts placed:")
                    print("    Texts:" + ";".join(res[0]['txt']) + "")
                    ws = re.sub(' +', ' ', (" ".join(res[0]['txt']).replace("\n", " "))).strip().split(" ")
                    print("    Words: #" +str(len(ws)) + " " + ";".join(ws) + "")
                    print("    Words bounding boxes: " + str(res[0]['wordBB'].shape) + "")
            else:
                print("    Failure: No text placed.")

            if len(res) > 0 and output_masks:
                ts = str(int(time() * 1000))

                # executed only if --output-masks flag is set
                prefix = MASKS_DIR + "/" + imname + ts

                imageio.imwrite(prefix + "_original.png", img)
                imageio.imwrite(prefix + "_with_text.png", res[0]['img'])

                # merge masks together:
                merged = reduce(lambda a, b: np.add(a, b), res[0]['masks'])
                # since we just added values of pixels, need to bring it back to 0..255 range.
                merged = np.divide(merged, len(res[0]['masks']))
                imageio.imwrite(prefix + "_mask.png", merged)

                # print bounding boxes
                f = open(prefix + "_bb.txt", "w+")
                bbs = res[0]['wordBB']
                boxes = np.swapaxes(bbs, 2, 0)
                words = re.sub(' +', ' ', ' '.join(res[0]['txt']).replace("\n", " ")).strip().split(" ")
                assert len(boxes) == len(words)
                for j in range(len(boxes)):
                    as_strings = np.char.mod('%f', boxes[j].flatten())
                    f.write(",".join(as_strings) + "," + words[j] + "\n")
                f.close()

            # visualize the output:
            if viz:
                # executed only if --viz flag is set
                for idict in res:
                    img_with_text = idict['img']
                    viz_textbb(1, img_with_text, [idict['wordBB']], alpha=1.0)
                    viz_masks(2, img_with_text, seg, depth, idict['labeled_region'])
                    # viz_regions(rgb.copy(),xyz,seg,regions['coeff'],regions['label'])
                    if i < INSTANCE_PER_IMAGE - 1:
                        raw_input(colorize(Color.BLUE, 'continue?', True))
                if 'q' in input(colorize(Color.RED, 'continue? (enter to continue, q to exit): ', True)):
                    break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue
    provider.close()
    out_db.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    parser.add_argument('--output-masks', action='store_true', dest='output_masks', default=False,
                        help='flag for turning on output of masks')
    parser.add_argument('--debug', action='store_true', dest='debug', default=False,
                        help='flag for turning on debug output')
    parser.add_argument("--data", type=str, dest='data_path', default=None,
                        help="absolute path to data directory containing images, segmaps and depths")
    args = parser.parse_args()
    main(viz=args.viz, debug=args.debug, output_masks=args.output_masks, data_path=args.data_path)
