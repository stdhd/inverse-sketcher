# HED batch processing script
# Modified from pix2pix scripts: https://github.com/phillipi/pix2pix/tree/master/scripts/edges
# Notice that this script processes images in batches instead of one-by-one, which is slightly faster.
# However, it will discard the last n images where 0 < n < batch_size.
# Originally modified from https://github.com/s9xie/hed/blob/master/examples/hed/HED-tutorial.ipynb
# Step 1: download the hed repo: https://github.com/s9xie/hed
# Step 2: download the models and protoxt, and put them under {caffe_root}/examples/hed/
# Step 3: put this script under {caffe_root}/examples/hed/
# Step 4: run the following script:
#       python batch_hed.py --images_dir=/data/to/path/photos/ --hed_mat_dir=/data/to/path/hed_mat_files/
# Step 5: run the MATLAB post-processing script "PostprocessHED.m"
# The code sometimes crashes after computation is done. Error looks like "Check failed: ... driver shutting down". You can just kill the job.
# For large images, it will produce gpu memory issue. Therefore, you better resize the images before running this script.

import numpy as np
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.io
import os
import cv2
import argparse
from time import time
from tqdm import trange, tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='batch proccesing: photos->edges')
    parser.add_argument('--images_dir', dest='images_dir',
                        default='sketch_processed',
                        help='directory to store input photos', type=str)
    parser.add_argument('--out_dir', dest='out_dir', default='edges_downsized',
                        help='directory to store output file',
                        type=str)
    args = parser.parse_args()
    return args


args = parse_args()

if not os.path.exists(args.out_dir):
    print('create output directory %s' % args.out_dir)
    os.makedirs(args.out_dir)
imgList = []
for subdir in os.listdir(args.images_dir):
    if not os.path.exists(os.path.join(args.out_dir, subdir)):
        print('create output directory %s' % os.path.join(args.out_dir, subdir))
        os.makedirs(os.path.join(args.out_dir, subdir))
    imgList.extend([os.path.join(subdir, img) for img in os.listdir(os.path.join(args.images_dir,subdir))])
#imgList = os.listdir(args.images_dir)
nImgs = len(imgList)
print('#images = %d' % nImgs)

for i in trange(nImgs):
    im = cv2.imread(os.path.join(args.images_dir, imgList[i]), cv2.IMREAD_COLOR)
    im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
    im = (im - np.min(im))*(np.max(im)/(np.max(im) - np.min(im)))
    name, ext = os.path.splitext(imgList[i])
    cv2.imwrite(os.path.join(args.out_dir, name + "_1.png"), im)
