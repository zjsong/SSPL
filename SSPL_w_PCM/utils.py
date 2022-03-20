"""
Common functions.
"""


import os
import cv2
import math
import shutil
import warnings
import numpy as np
import xml.etree.ElementTree as ET
import scipy.io.wavfile as wavfile

from PIL import Image
from sklearn import metrics

import torch

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def makedirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
            print('Removed existing directory...')
        else:
            return
    os.makedirs(path)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()

    def sum_value(self):
        if self.sum is None:
            return 0.
        else:
            return self.sum.tolist()


def plot_loss_metrics(path, history):

    # plot ssl loss
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['loss'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['loss'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss_ssl.png'), dpi=200)
    plt.close('all')

    # plot validation performance
    fig = plt.figure()
    plt.plot(history['val']['epoch'], history['val']['ciou'],
             color='r', label='cIoU')
    plt.plot(history['val']['epoch'], history['val']['auc'],
             color='g', label='AUC')
    plt.legend()
    fig.savefig(os.path.join(path, 'ciou_auc.png'), dpi=200)
    plt.close('all')


def save_visual(video_id, img, orig_sim_map, args):

    # frame
    img = np.uint8(normalize_img(img.transpose(1, 2, 0)) * 255)
    img = Image.fromarray(img)

    # heat map
    orig_sim_map = normalize_img(orig_sim_map)
    orig_sim_map = matrix2heatmap(orig_sim_map * 255)
    orig_sim_map = Image.fromarray(orig_sim_map)
    orig_sim_map = orig_sim_map.resize((img.size[0], img.size[1]), resample=Image.BILINEAR)
    img_orig_sim_map = Image.blend(img, orig_sim_map, alpha=0.5)
    img_orig_sim_map.save(os.path.join(args.vis, video_id + '_img_orig_sim_map.png'))


def save_visual_eval(video_id, img, sound, orig_sim_map, orig_sim_map_rsf, args):

    # frame
    img = np.uint8(normalize_img(img.transpose(1, 2, 0)) * 255)
    img = Image.fromarray(img)

    # sound
    wavfile.write(os.path.join(args.vis, video_id + '_sound.wav'), args.audRate, sound)

    # localization map
    orig_sim_map = normalize_img(orig_sim_map)
    orig_sim_map = matrix2heatmap(orig_sim_map * 255)
    orig_sim_map = Image.fromarray(orig_sim_map)
    orig_sim_map = orig_sim_map.resize((img.size[0], img.size[1]), resample=Image.BILINEAR)
    img_orig_sim_map = Image.blend(img, orig_sim_map, alpha=0.5)
    img_orig_sim_map.save(os.path.join(args.vis, video_id + '_img_orig_sim_map.png'))

    orig_sim_map_rsf = normalize_img(orig_sim_map_rsf)
    orig_sim_map_rsf = matrix2heatmap(orig_sim_map_rsf * 255)
    orig_sim_map_rsf = Image.fromarray(orig_sim_map_rsf)
    orig_sim_map_rsf = orig_sim_map_rsf.resize((img.size[0], img.size[1]), resample=Image.BILINEAR)
    img_orig_sim_map_rsf = Image.blend(img, orig_sim_map_rsf, alpha=0.5)
    img_orig_sim_map_rsf.save(os.path.join(args.vis, video_id + '_img_orig_sim_map_rsf.png'))


def normalize_img(image):
    """
    Normalize the given image (H x W) or (H x W x C) to [0, 1].
    """
    if len(image.shape) == 2:
        image = (image - np.min(image)) / np.ptp(image)

    elif len(image.shape) == 3:
        image[:, :, 0] = (image[:, :, 0] - np.min(image[:, :, 0])) / np.ptp(image[:, :, 0])
        image[:, :, 1] = (image[:, :, 1] - np.min(image[:, :, 1])) / np.ptp(image[:, :, 1])
        image[:, :, 2] = (image[:, :, 2] - np.min(image[:, :, 2])) / np.ptp(image[:, :, 2])

    return image


def matrix2heatmap(image):
    """
    Input image should be a 2D ndarray with values in [0, 255].
    """
    image = image.astype(np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    image = image[:, :, ::-1]   # flip the color from BGR (in OpenCV) to RGB (in Image) for display
    return image


def normalize_audio(audio_data):
    EPS = 1e-3
    min_data = audio_data.min()
    audio_data -= min_data
    max_data = audio_data.max()
    audio_data /= max_data + EPS
    audio_data -= 0.5
    audio_data *= 2
    return audio_data


def testset_gt(args, name):
    if args.testset == 'flickr':
        gt = ET.parse(args.data_path + 'SoundNet_Flickr/5k_labeled/Annotations/' + '%s.xml' % name).getroot()
        gt_map = np.zeros([args.imgSize, args.imgSize])
        bboxs = []
        for child in gt:
            for childs in child:
                bbox = []
                if childs.tag == 'bbox':
                    for index, ch in enumerate(childs):
                        if index == 0:
                            continue
                        bbox.append(int(args.imgSize * int(ch.text) / 256))
                bboxs.append(bbox)
        for item_ in bboxs:
            temp = np.zeros([args.imgSize, args.imgSize])
            (xmin, ymin, xmax, ymax) = item_[0], item_[1], item_[2], item_[3]
            temp[item_[1]:item_[3], item_[0]:item_[2]] = 1
            gt_map += temp
        gt_map /= 2
        gt_map[gt_map > 1] = 1

    elif args.testset == 'vggss':
        gt = args.gt_all[name]
        gt_map = np.zeros([args.imgSize, args.imgSize])
        for item_ in gt:
            item_ = list(map(lambda x: int(args.imgSize * max(x, 0)), item_))
            temp = np.zeros([args.imgSize, args.imgSize])
            (xmin, ymin, xmax, ymax) = item_[0], item_[1], item_[2], item_[3]
            temp[ymin:ymax, xmin:xmax] = 1
            gt_map += temp
        gt_map[gt_map > 0] = 1
    return gt_map
