import os
import h5py
import random
import numpy as np

import torch

from .base import BaseDataset
from models.torchvggish.torchvggish import vggish_input


class VideoDataset(BaseDataset):
    def __init__(self, args, mode):
        super(VideoDataset, self).__init__(args, mode)

    def __getitem__(self, index):

        # extract data
        if self.mode == 'train':

            if self.num_data == 10000:
                
                if self.trainset == 'flickr':
                    data_path = self.args.data_path + 'SoundNet_Flickr/10k_unlabeled/'
                elif self.trainset == 'vggsound':
                    data_path = self.args.data_path + 'VGG-Sound/10k_unlabeled/'
                
                # frame
                with h5py.File(data_path + 'h5py_train_frames_10k.h5', 'r') as hf:
                    frame = hf['train_frames'][index, :, :, :]   # (256, 256, 3)
                    frame_list = self._load_frame(frame=frame)
                # audio
                with h5py.File(data_path + 'h5py_train_audios_10k.h5', 'r') as hf:
                    audio = hf['train_audios'][index, :]   # (48000,)
                # log-mel spectrogram
                with h5py.File(data_path + 'h5py_train_spects_10k.h5', 'r') as hf:
                    spect = hf['train_spects'][index, :, :, :]
                    spect = torch.tensor(spect)[:, None, :, :].float()   # torch.Size([3, 1, 96, 64])

            elif self.num_data == 144000:
                
                if self.trainset == 'flickr':
                    data_path = self.args.data_path + 'SoundNet_Flickr/144k_unlabeled/'
                elif self.trainset == 'vggsound':
                    data_path = self.args.data_path + 'VGG-Sound/144k_unlabeled/'
                
                id_h5 = index // 14400   # totally 10 h5 files, each file has 14400 samples
                id_data = index % 14400
                # frame
                with h5py.File(data_path + 'h5py_train_frames_144k_' + str(id_h5 + 1) + '.h5', 'r') as hf:
                    frame = hf['train_frames'][id_data, :, :, :]  # (256, 256, 3)
                    frame_list = self._load_frame(frame=frame)
                # audio
                with h5py.File(data_path + 'h5py_train_audios_144k_' + str(id_h5 + 1) + '.h5', 'r') as hf:
                    audio = hf['train_audios'][id_data, :]  # (48000,)
                # log-mel spectrogram
                with h5py.File(data_path + 'h5py_train_spects_144k_' + str(id_h5 + 1) + '.h5', 'r') as hf:
                    spect = hf['train_spects'][id_data, :, :, :]
                    spect = torch.tensor(spect)[:, None, :, :].float()  # torch.Size([3, 1, 96, 64])

        else:
            
            data_id = str(self.data_ids.iat[index, 0])
            
            if self.testset == 'flickr':
                frame_path = self.args.data_path + 'SoundNet_Flickr/5k_labeled/Data/frames/' + data_id + '.jpg'
                audio_path = self.args.data_path + 'SoundNet_Flickr/5k_labeled/Data/audio/' + data_id + '.wav'

                # frame
                frame_list = self._load_frame(path=frame_path)
                # audio
                audio = self._load_audio(path=audio_path)
                # log-mel spectrogram
                spect = vggish_input.waveform_to_examples(audio, sample_rate=self.audRate, return_tensor=False)
                spect = torch.tensor(spect)[:, None, :, :].float()
            
            elif self.testset == 'vggss':
                data_path = self.args.data_path + 'VGG-Sound/5k_labeled/Data/'
                # frame
                with h5py.File(data_path + 'h5py_test_frames.h5', 'r') as hf:
                    frame = hf['test_frames'][index, :, :, :]   # (256, 256, 3)
                    frame_list = self._load_frame(frame=frame)
                # audio
                with h5py.File(data_path + 'h5py_test_audios.h5', 'r') as hf:
                    audio = hf['test_audios'][index, :]  # (48000,)
                # log-mel spectrogram
                with h5py.File(data_path + 'h5py_test_spects.h5', 'r') as hf:
                    spect = hf['test_spects'][index, :, :, :]
                    spect = torch.tensor(spect)[:, None, :, :].float()  # torch.Size([3, 1, 96, 64])

        # output
        output_dict = {'frame_view1': frame_list[0], 'frame_view2': frame_list[1], 'sound': audio, 'spect': spect}
        if self.mode == 'val' or self.mode == 'test':
            output_dict['data_id'] = data_id

        return output_dict
