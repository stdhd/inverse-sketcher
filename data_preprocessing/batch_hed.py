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
from time import time, sleep
from tqdm import trange, tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='batch proccesing: photos->edges')
    parser.add_argument('--caffe_root', dest='caffe_root', help='caffe root',
                        default='/home/russell/caffe-segnet/caffe', type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', help='caffemodel',
                        default='/home/russell/hed/examples/hed/hed_pretrained_bsds.caffemodel',
                        type=str)
    parser.add_argument('--prototxt', dest='prototxt', help='caffe prototxt file',
                        default='/home/russell/hed/examples/hed/deploy.prototxt', type=str)
    parser.add_argument('--images_dir', dest='images_dir',
                        default='images',
                        help='directory to store input photos', type=str)
    parser.add_argument('--hed_mat_dir', dest='hed_mat_dir', default='edges',
                        help='directory to store output hed edges in mat file',
                        type=str)
    parser.add_argument('--border', dest='border', help='padding border', type=int, default=64  )
    parser.add_argument('--gpu_id', dest='gpu_id', help='gpu id', type=int, default=0)
    args = parser.parse_args()
    return args


args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))
## Make sure that caffe is on the python path:
caffe_root = args.caffe_root  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))

# Suppress output
os.environ['GLOG_minloglevel'] = '2'

import caffe
import scipy.io as sio

if not os.path.exists(args.hed_mat_dir):
    print('create output directory %s' % args.hed_mat_dir)
    os.makedirs(args.hed_mat_dir)
imgList = []
for subdir in os.listdir(args.images_dir):
    if not os.path.exists(os.path.join(args.hed_mat_dir, subdir)):
        print('create output directory %s' % os.path.join(args.hed_mat_dir, subdir))
        os.makedirs(os.path.join(args.hed_mat_dir, subdir))
    imgList.extend([os.path.join(subdir, img) for img in os.listdir(os.path.join(args.images_dir,subdir))])
#imgList = os.listdir(args.images_dir)
nImgs = len(imgList)
print('#images = %d' % nImgs)

caffe.set_mode_gpu()
caffe.set_device(args.gpu_id)
# load net
net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
# pad border
border = args.border

# Time counter
prev_time = float("-inf")
curr_time = float("-inf")
image_list = []
batch_size = 32

print(imgList[0])

for i in trange(nImgs):
    if i % 4000 == 0:
        tqdm.write('processing image %d/%d' % (i, nImgs))
        curr_time = time()
        elapsed = curr_time - prev_time
        tqdm.write(
            "Now at iteration %d. Elapsed time: %.5fs. Average time: %.5fs/iter" % (i, elapsed, elapsed / 500.))
        prev_time = curr_time

    if i % batch_size == 0 and i > 0:
        in_ = np.concatenate(image_list, axis=0)

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(*in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        fuse = np.squeeze(net.blobs['sigmoid-fuse'].data)
        # get rid of the border
        #fuse = fuse[:, border:-border, border:-border]
        fuse = fuse[:, int(border*1.5):-int(border*0.5), int(border*1.5):-int(border*0.5)]
        #cv2.imshow("...", fuse[2])
        #cv2.waitKey(3)

        #sleep(2)

        for j in range(batch_size):
            # save hed file to the disk
            name, ext = os.path.splitext(imgList[i - batch_size + j])
            sio.savemat(os.path.join(args.hed_mat_dir, name + '.mat'), {'predict': fuse[j]})
            #cv2.imwrite(os.path.join(args.hed_mat_dir, name + "_1.png"), cv2.resize((np.asarray(fuse[j]) - np.min(fuse[j]))/(np.max(fuse[j]) - np.min(fuse[j]))*255, (64, 64)))
        image_list = []

    im = cv2.imread(os.path.join(args.images_dir, imgList[i]), cv2.IMREAD_COLOR)
    im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)

    in_ = im.astype(np.float32)
    in_ = np.pad(in_, ((border, border), (border, border), (0, 0)), 'reflect')

    in_ = in_[:, :, ::-1]
    in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
    in_ = np.expand_dims(in_.transpose((2, 0, 1)), axis=0)
    #print(type(in_), in_.shape)
    #cv2.imshow("...", in_[0][:, border:-border, border:-border].transpose(1, 2, 0))

    image_list.append(in_)
