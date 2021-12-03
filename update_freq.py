#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import pickle

import configuration


def update_freq(data_path):
    cnt = 0
    filename = data_path +"/" + configuration.text_soruce
    with open(filename) as f:
        c = Counter()
        for x in f:
            c += Counter(x.strip())
            cnt += len(x.strip())
            # print c
    print(cnt)
    for key in c:
        c[key] = float(c[key]) / cnt
        print(key, c[key])
    d = dict(c)
    # print d
    with open("{}/{}".format(data_path,configuration.char_freq_path), 'wb') as f:
        pickle.dump(d, f)

import argparse
parser = argparse.ArgumentParser(description='invert font size')
parser.add_argument('--lang', default='ENG',
                    help='Select language : ENG/HI')
parser.add_argument("--data_path", default="data/")

args = parser.parse_args()
configuration.char_freq_path = 'models/{}/char_freq.cp'.format(args.lang)
configuration.font_px2pt = 'models/{}/font_px2pt.cp'.format(args.lang)
configuration.text_soruce = "newsgroup/newsgroup_{}.txt".format(args.lang)

update_freq(args.data_path)