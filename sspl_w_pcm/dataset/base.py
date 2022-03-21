"""
A base class for constructing PyTorch AudioVisual dataset.
"""


import random
import librosa
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch.utils.data as torchdata
import torchvision.transforms.functional as TF
from torchvision import transforms
from utils import normalize_audio


class BaseDataset(torchdata.Dataset):
    def __init__(self, args, mode='train'):
        # params
        self.args = args
        self.mode = mode
        self.seed = args.seed
        random.seed(self.seed)

        self.imgSize = args.imgSize
        self.audRate = args.audRate
        self.audSec = args.audSec  # 3s
        self.audLen = args.audRate * args.audSec
        self.trainset = args.trainset
        self.testset = args.testset

        # initialize visual and audio transform
        self._init_vtransform()
        self._init_atransform()

        if self.mode == 'train':
            self.num_data = args.num_train
            print('number of training samples: ', self.num_data)

        elif self.mode == 'test':
            if self.testset == 'flickr':
                data_path = args.data_path + 'SoundNet_Flickr/flickr_test249_in5k.csv'
            elif self.testset == 'vggss':
                data_path = args.data_path + 'VGG-Sound/vggss_test_4692.csv'

            self.data_ids = pd.read_csv(data_path, header=None, sep=',')
            self.num_data = self.data_ids.shape[0]
            print('number of test samples: ', self.num_data)

    def __len__(self):
        return self.num_data

    # video frame transform funcs
    def _init_vtransform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    # audio waveform transform funcs
    def _init_atransform(self):
        self.aud_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[12.0])])

    def _load_frame(self, frame=None, path=None, num_trans=2):
        """
        Generate num_trans samples with augmentation.
        """
        if frame is not None:
            frame = Image.fromarray(np.uint8(frame))
        else:
            frame = Image.open(path).convert('RGB')

        return [self.img_transform(frame) for _ in range(num_trans)]

    def _load_audio(self, path):
        """
        Load wav file.
        """

        # load audio
        audio_np, rate = librosa.load(path, sr=self.audRate, mono=True)

        curr_audio_length = audio_np.shape[0]
        if curr_audio_length < self.audLen:
            n = int(self.audLen / curr_audio_length) + 1
            audio_np = np.tile(audio_np, n)
            curr_audio_length = audio_np.shape[0]

        start_sample = int(curr_audio_length / 2) - int(self.audLen / 2)
        audio_np = normalize_audio(audio_np[start_sample:start_sample + self.audLen])

        return audio_np


class MyRotationTransform(object):
    """
    Rotate by one of the given angles.
    """

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

