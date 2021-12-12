# Author: Ankush Gupta
# Date: 2015

"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division

import math
import os.path
import random

import h5py
import pandas as pd
import pickle5

import configuration
from common import *
from math import floor, ceil

import numpy as np
import cv2

import lmdb


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    tr, br = rightMost

    x1, y1 = bl
    x2, y2 = br
    w1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    x1, y1 = tl
    x2, y2 = tr
    w2 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # width of box
    w = max(w1, w2)

    x1, y1 = tl
    x2, y2 = bl
    h1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    x1, y1 = tr
    x2, y2 = br
    h2 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    h = max(h1, h2)  # height of box

    return np.array([tl, tr, br, bl], dtype="float32"), int(w), int(h)


def convert_floating_coordinates_to_int(bb):
    bb[0][0] = floor(bb[0][0])
    bb[0][1] = ceil(bb[0][1])
    bb[0][2] = ceil(bb[0][2])
    bb[0][3] = floor(bb[0][3])
    bb[1][0] = floor(bb[1][0])
    bb[1][1] = floor(bb[1][1])
    bb[1][2] = ceil(bb[1][2])
    bb[1][3] = ceil(bb[1][3])
    bb = np.where(bb < 0, 0, bb)
    return bb.astype(int)


def get_string_representation_of_bbox(bb):
    return "{},{},{},{},{},{},{},{}".format(bb[0][0], bb[0][1], bb[1][0], bb[1][1], bb[2][0], bb[2][1], bb[3][0],
                                            bb[3][1])


def crop_words(text_im, bb, image_name=None, text=None):
    """
	text_im : image containing text
	charBB_list : list of 2x4xn_i bounding-box matrices
	wordBB : 2x4xm matrix of word coordinates
	"""

    # bb = convert_floating_coordinates_to_int(wordBB).T
    bb, width, height = order_points(bb)
    src_pts = bb.astype("float32")
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(text_im, M, (width, height))
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    # print(bb)

    min_x = min(bb[0][0], bb[1][0], bb[2][0], bb[3][0])
    max_x = max(bb[0][0], bb[1][0], bb[2][0], bb[3][0])

    min_y = min(bb[0][1], bb[1][1], bb[2][1], bb[3][1])
    max_y = max(bb[0][1], bb[1][1], bb[2][1], bb[3][1])

    # print(min_x, max_x, min_y, max_y)

    # print(min_x, max_x, min_y, max_y)

    unwarped = text_im[int(min_y): int(max_y), int(min_x): int(max_x)]
    # print("Unwarped", unwarped)

    # bouding boxes are cropped using 2 techqniques, warped and unwarped both represent different challenges for text
    # recognition model.

    return warped, unwarped


def create_recognition_dataset_from_db_file(db_fname):
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))
    for k in dsets:
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']
        font = db['data'][k].attrs['font']

        j = k.replace(".", "_")
        crop_words(rgb, wordBB, image_name=j, text=txt)
        print("image name        : ", colorize(Color.RED, k, bold=True))
        print("  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1]))
        print("  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1]))
        print("  ** text         : ", colorize(Color.GREEN, txt))
        print('  ** font         : ', colorize(Color.GREEN, font))


def writeCache(env, cache):
    # print("Writing to LMDB")
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            v = np.array((v), dtype=object)
            # print(v[1], v[2], v[3])
            txn.put(k.encode("ascii"), pickle5.dumps(v))


def create_detection_dataset(input_path, output_path, gt_file):
    """
	Create LMDB dataset for detection.
	ARGS:
		input_path  : input folder path for images
		output_path : LMDB output path
		gtFile     : list of image path and label
	"""

    # df = pd.read_csv(gt_file, sep="\t")
    # print(df)

    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    lines = open(gt_file).readlines()

    print("Starting...")
    cnt = 1;

    for line in lines:
        img_path, word_bb, text, font = line.split("\t")
        img_name = os.path.basename(img_path)
        img_path = os.path.join(input_path, img_name)

        # print(line)
        # print(img_name)

        with open(img_path, 'rb') as f:
            img = cv2.imread(img_path)
            img_bin = f.read()

        # print(word_bb)

        word_bb = word_bb.split(",")
        # print(word_bb)

        word_bb = np.array([int(float(bb)) for bb in word_bb]).reshape((4, 2))
        # print(word_bb)

        key = img_name
        # print(key)

        if key in cache:
            word_bb_list = cache[key][1]
            text_list = cache[key][2]
            font_list = cache[key][3]

            word_bb_list.append(word_bb)
            text_list.append(text)
            font_list.append(font[:-1])

            cache[key] = [cache[key][0], word_bb_list, text_list, font_list]
        # print(word_bb_list, text_list, font_list)

        else:
            cache[key] = [img_bin, [word_bb], [text], [font[:-1]]]

        if cnt % 10000 == 0:
            print("Done " + str(cnt) + "/" + str(len(lines)))

        cnt += 1

    writeCache(env, cache)
    print("Done")

def create_detection_dataset_combined(input_path, output_path):
    for file in os.listdir(input_path):
        if file.endswith(".txt"):
            print("Processing ", file)
            create_detection_dataset(input_path, output_path, os.path.join(input_path, file))

def create_recognition_dataset_warped_unwarped_from_detection_dataset(images_path, input_path, output_path):
    """
	Create LMDB dataset for detection.
	ARGS:
		input_path  : input lmdb detection dataset
		output_path : LMDB output path
	"""

    val_output_path = os.path.join(output_path, "val")
    train_output_path = os.path.join(output_path, "train")

    os.makedirs(val_output_path, exist_ok=True)
    os.makedirs(train_output_path, exist_ok=True)

    print("Fetching Keys...")
    keys = []
    cursor = lmdb.open(input_path, readonly=True).begin().cursor()
    for key, value in cursor:
        key = key.decode()
        keys.append(key)

    print("Done")

    print()
    print("Splitting train and val keys")
    val_keys = random.sample(keys, (int)(0.1 * len(keys)))
    train_keys = []
    for key in keys:
        if key not in val_keys:
            train_keys.append(key)

    # print("Val Images", val_keys)
    # print("Train Images", train_keys)
    print("Done")

    ##### TRAINING DATASET #####
    print()
    print("Start creating training dataset...")

    env = lmdb.open(train_output_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    txn = lmdb.open(input_path, readonly=True).begin()
    for key in train_keys:
        value = pickle5.loads(txn.get(key.encode("ascii")))

        img_name = key
        img_path = os.path.join(images_path, img_name)

        # print(img_path)

        with open(img_path, 'rb') as f:
            img = cv2.imread(img_path)

        # print("Current Image:", img_name)

        for i, (bb, text, font) in enumerate(zip(value[1], value[2], value[3])):
            # print(i)

            warped, unwarped = crop_words(img, bb, image_name=img_name, text=text)

            warped_img_name = img_name[:-4] + "_" + str(i) + "_" + "warped.jpg";
            unwarped_img_name = img_name[:-4] + "_" + str(i) + "_" + "unwarped.jpg";

            try:
                cache[warped_img_name] = [warped, bb, text, font]
                cv2.imwrite(os.path.join(train_output_path, warped_img_name), warped)
            except:
                print(warped_img_name + " is empty!")

            try:
                cache[unwarped_img_name] = [unwarped, bb, text, font]
                cv2.imwrite(os.path.join(train_output_path, unwarped_img_name), unwarped)
            except:
                print(unwarped_img_name + " is empty!")

            if cnt % 10 == 0:
                writeCache(env, cache)
                cache = {}
                print('Done ', cnt)

            cnt += 1

    writeCache(env, cache)
    print("Done")

    ##### VALIDATION DATASET #####
    print()
    print("Start creating validation dataset...")

    env = lmdb.open(val_output_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    txn = lmdb.open(input_path, readonly=True).begin()
    for key in val_keys:
        value = pickle5.loads(txn.get(key.encode("ascii")))

        img_name = key
        img_path = os.path.join(images_path, img_name)

        # print(img_path)

        with open(img_path, 'rb') as f:
            img = cv2.imread(img_path)

        # print("Current Image:", img_name)

        for i, (bb, text, font) in enumerate(zip(value[1], value[2], value[3])):
            # print(i)

            warped, unwarped = crop_words(img, bb, image_name=img_name, text=text)

            warped_img_name = img_name[:-4] + "_" + str(i) + "_" + "warped.jpg";
            unwarped_img_name = img_name[:-4] + "_" + str(i) + "_" + "unwarped.jpg";

            try:
                cache[warped_img_name] = [warped, bb, text, font]
                cv2.imwrite(os.path.join(val_output_path, warped_img_name), warped)
            except:
                print(warped_img_name + " is empty!")

            try:
                cache[unwarped_img_name] = [unwarped, bb, text, font]
            except:
                print(unwarped_img_name + " is empty!")

            if cnt % 10 == 0:
                writeCache(env, cache)
                cache = {}
                print('Done ', cnt)

            cnt += 1

    writeCache(env, cache)
    print("Done")

def create_recognition_dataset_warped_unwarped(input_path, output_path, gt_file):
    """
	Create LMDB dataset for detection.
	ARGS:
		input_path  : input lmdb detection dataset
		output_path : LMDB output path
	"""

    val_output_path = os.path.join(output_path, "val")
    train_output_path = os.path.join(output_path, "train")

    os.makedirs(val_output_path, exist_ok=True)
    os.makedirs(train_output_path, exist_ok=True)

    print("Fetching Keys...")
    lines = open(gt_file).readlines()
    print("Done")

    print()
    print("Splitting train and val keys")
    val_keys = random.sample(lines, (int)(0.1 * len(lines)))
    train_keys = []
    for key in lines:
        if key not in val_keys:
            train_keys.append(key)

    # print(len(lines), len(val_keys), len(train_keys))
    # print("Val Images", val_keys)
    # print("Train Images", train_keys)
    print("Done")

    ##### TRAINING DATASET #####
    print()
    print("Start creating training dataset...")

    env = lmdb.open(train_output_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    for i, key in enumerate(train_keys):

        img_path, word_bb, text, font = key.split("\t")
        img_name = os.path.basename(img_path)
        img_path = os.path.join(input_path, img_name)

        # print(line)
        # print(img_name)

        with open(img_path, 'rb') as f:
            img = cv2.imread(img_path)
            img_bin = f.read()

        # print(word_bb)

        word_bb = word_bb.split(",")
        # print(word_bb)

        word_bb = np.array([int(float(bb)) for bb in word_bb]).reshape((4, 2))
        # print(word_bb)

        key = img_name
        # print(key)

        warped, unwarped = crop_words(img, word_bb, image_name=img_name, text=text)

        warped_img_name = img_name[:-4] + "_" + str(i) + "_" + "warped.jpg";
        unwarped_img_name = img_name[:-4] + "_" + str(i) + "_" + "unwarped.jpg";

        try:
            cache[warped_img_name] = [warped.tobytes(), word_bb, text, font]
            # cv2.imwrite(os.path.join(train_output_path, warped_img_name), warped)
        except:
            print(warped_img_name + " is empty!")

        try:
            cache[unwarped_img_name] = [unwarped.tobytes(), word_bb, text, font]
            # cv2.imwrite(os.path.join(train_output_path, unwarped_img_name), unwarped)
        except:
            print(unwarped_img_name + " is empty!")

        if cnt % 10 == 0:
            writeCache(env, cache)
            cache = {}
            print('Done ' + str(cnt) + ' /' + str(len(train_keys)))

        cnt += 1

    writeCache(env, cache)
    print("Done")

    ##### VALIDATION DATASET #####
    print()
    print("Start creating validation dataset...")

    env = lmdb.open(val_output_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    for i, key in enumerate(val_keys):

        img_path, word_bb, text, font = key.split("\t")
        img_name = os.path.basename(img_path)
        img_path = os.path.join(input_path, img_name)

        # print(line)
        # print(img_name)

        with open(img_path, 'rb') as f:
            img = cv2.imread(img_path)
            img_bin = f.read()

        # print(word_bb)

        word_bb = word_bb.split(",")
        # print(word_bb)

        word_bb = np.array([int(float(bb)) for bb in word_bb]).reshape((4, 2))
        # print(word_bb)

        key = img_name
        # print(key)

        warped, unwarped = crop_words(img, word_bb, image_name=img_name, text=text)

        warped_img_name = img_name[:-4] + "_" + str(i) + "_" + "warped.jpg";
        unwarped_img_name = img_name[:-4] + "_" + str(i) + "_" + "unwarped.jpg";

        try:
            cache[warped_img_name] = [warped.tobytes(), word_bb, text, font]
            # cv2.imwrite(os.path.join(val_output_path, warped_img_name), warped)
        except:
            print(warped_img_name + " is empty!")

        try:
            cache[unwarped_img_name] = [unwarped.tobytes(), word_bb, text, font]
            # cv2.imwrite(os.path.join(val_output_path, unwarped_img_name), unwarped)
        except:
            print(unwarped_img_name + " is empty!")

        if cnt % 10 == 0:
            writeCache(env, cache)
            cache = {}
            print('Done ' + str(cnt) + ' /' + str(len(val_keys)))

        cnt += 1

    writeCache(env, cache)
    print("Done")

def create_recognition_dataset_warped_unwarped_combined(input_path, output_path):
    for file in os.listdir(input_path):
        if file.endswith(".txt"):
            print()
            print("Processing ", file)
            create_recognition_dataset_warped_unwarped(input_path, output_path, os.path.join(input_path, file))

def create_recognition_dataset_from_detection_datset_folder(path, gt_file):
    lines = gt_file.readlines()
    for line in lines:
        img_path, word_bb, text, font = line.split("\t")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        word_bb = word_bb.split(",")
        word_bb = np.array([int(bb) for bb in word_bb]).reshape((4, 2))
        img_name = os.path.basename(img_path)
        crop_words(img, word_bb, image_name=img_name, text=text)
        print("image name        : ", colorize(Color.RED, bold=True))
        print("  ** text         : ", colorize(Color.GREEN, text))
        print('  ** font         : ', colorize(Color.GREEN, font))


def main(db_fname):
    create_recognition_dataset_from_detection_datset_folder()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='crop images and create recogntion dataset')

    parser.add_argument('--lang', default='ENG',
                        help='Select language : ENG/HI')

    parser.add_argument('--path', default='./output',
                        help='Select language : ENG/HI')

    parser.add_argument('--gt_file', default='./output/gt.txt',
                        help='Select language : ENG/HI')

    args = parser.parse_args()

    configuration.lang = args.lang

    # main('./SynthText_{}.h5'.format(configuration.lang))
    # create_detection_dataset_combined("/home/manideep/Desktop/Indic OCR/hindi-sample/output", "hindi-sample-detection")
    # create_recognition_dataset_warped_unwarped_combined("/home/manideep/Desktop/Indic OCR/hindi-sample/output", "hindi-sample-recognition")