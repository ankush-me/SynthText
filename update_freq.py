#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import pickle

import configuration

cnt = 0
filename = configuration.text_soruce
with open(filename) as f:
    c = Counter()
    for x in f:
        c += Counter(x.strip())
        cnt += len(x.strip())
        # print c
print(cnt)

for key in c:
    c[key] = float(c[key]) / cnt
    print (key, c[key])

d = dict(c)
# print d
with open("char_freq.cp", 'wb') as f:
    pickle.dump(d, f)