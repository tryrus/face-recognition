# conding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import csv
import os
from mtcnn_net import *


model = '20170512-110547/20170512-110547.pb'
db_path = 'db'
image_size = 160
margin = 44

if not os.path.exists(db_path):
    print('Database path is not existed!')
folders = sorted(glob.glob(os.path.join(db_path, '*')))


features_all = []
for name in folders:
    #print(name)
    images = glob.glob(os.path.join(name, '*.jpg'))
    images = load_and_align_data(images, image_size)
    feature = compute_feature(model, images)
    features_all.append(feature)

with open("data/features_all.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for n in range(len(features_all)):
        writer.writerow(features_all[n])