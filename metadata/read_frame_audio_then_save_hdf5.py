"""
Read frame and audio data using given video ids, and then save as hdf5, respectively.
"""


import os
import cv2
import h5py
import glob
import pandas as pd
import numpy as np
import librosa
from pydub import AudioSegment
from utils import normalize_audio
from models.torchvggish.torchvggish import vggish_input


# -----------------------------------------------
# raw training data paths and hyper-parameters
# -----------------------------------------------
root_data = "Path_to_SoundNet_Flickr/"
path_video_train_all = root_data + 'all_unlabeled/flickr_train_videos_1999751.csv'
video_train_all_paths = pd.read_csv(path_video_train_all, header=None, sep=',')   # shape: (1999751, 1)
num_train_all = video_train_all_paths.shape[0]   # 1,999,751
sample_size = 10000   # {10000, 144000, 1999751} for training
audio_sample_rate = 16000
audio_duration = 3
audio_length = audio_sample_rate * audio_duration


# -----------------------------------------------
# extract paths and data
# -----------------------------------------------
video_train_sample_paths = [0] * sample_size
sample_frames = np.zeros((sample_size, 256, 256, 3), dtype='float32')
sample_audios = np.zeros((sample_size, audio_length), dtype='float32')
sample_spects = np.zeros((sample_size, audio_duration, 96, 64), dtype='float32')   # (sample_size, audio_duration, 96, 64) for VGGish
cnt_saved_data = 0
for i in range(num_train_all):

    path = video_train_all_paths.iat[i, 0]   # path: e.g., 'videos2/4/8/1/6/6/3/4/4/6248166344.mp4'

    # frame (jpg)
    frames_path = root_data + 'frames/' + path + '/'
    all_jpgs = glob.glob1(frames_path, '*.jpg')
    if len(all_jpgs) == 0:   # skip empty folder
        print('Path does not exist, or there is no frame: ', path)
        continue

    # audio (mp3)
    path_ = path.rsplit('/', 1)[0]   # e.g., 'videos2/4/8/1/6/6/3/4/4'
    audio_path = root_data + 'mp3/' + path_ + '/'
    audio = glob.glob1(audio_path, '*.mp3')
    if len(audio) == 0:      # skip empty folder
        print('Path does not exist, or there is no audio: ', path_)
        continue

    video_train_sample_paths[cnt_saved_data] = path

    frame_path = frames_path + all_jpgs[int(len(all_jpgs) / 2)]
    audio_path_mp3 = audio_path + audio[0]

    # record one frame image for one video
    frame = cv2.imread(frame_path)
    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
    sample_frames[cnt_saved_data, :, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # record 3s sound spectrogram for one video
    # convert mp3 to wav
    audio_path_wav = audio_path + audio[0].rsplit('.', 1)[0] + '.wav'
    sound = AudioSegment.from_mp3(audio_path_mp3)
    sound.export(audio_path_wav, format='wav')
    # read wav with given duration
    audio_np, rate = librosa.load(audio_path_wav, sr=audio_sample_rate, mono=True)
    curr_audio_length = audio_np.shape[0]
    if curr_audio_length < audio_length:
        n = int(audio_length / curr_audio_length) + 1
        audio_np = np.tile(audio_np, n)
        curr_audio_length = audio_np.shape[0]
    start_sample = int(curr_audio_length / 2) - int(audio_length / 2)
    sample_audios[cnt_saved_data, :] = normalize_audio(audio_np[start_sample:start_sample + audio_length])
    # log-mel spectrogram
    sample_spects[cnt_saved_data, :, :, :] = vggish_input.waveform_to_examples(sample_audios[cnt_saved_data, :],
                                                                               sample_rate=audio_sample_rate,
                                                                               return_tensor=False)

    cnt_saved_data += 1
    if cnt_saved_data % 1000 == 0:
        print('{} sample data have been extracted.'.format(cnt_saved_data))
    if cnt_saved_data == sample_size:
        print('All {} sample data have been extracted.'.format(cnt_saved_data))
        break

# -----------------------------------------------
# save paths and data
# -----------------------------------------------
# save paths into csv
if sample_size == 10000:
    saved_train_sample_paths = root_data + '10k_unlabeled/flickr_train_videos_10k.csv'
elif sample_size == 144000:
    saved_train_sample_paths = root_data + '144k_unlabeled/flickr_train_videos_144k.csv'
video_train_sample_paths = pd.DataFrame(data=video_train_sample_paths)
video_train_sample_paths.to_csv(saved_train_sample_paths, header=None, index=0)

# save data into hdf5
if sample_size == 10000:
    h5py_path_frames = root_data + '10k_unlabeled/h5py_train_frames_10k.h5'
    h5py_path_audios = root_data + '10k_unlabeled/h5py_train_audios_10k.h5'
    h5py_path_spects = root_data + '10k_unlabeled/h5py_train_spects_10k.h5'
elif sample_size == 144000:
    h5py_path_frames = root_data + '144k_unlabeled/h5py_train_frames_144k.h5'
    h5py_path_audios = root_data + '144k_unlabeled/h5py_train_audios_144k.h5'
    h5py_path_spects = root_data + '144k_unlabeled/h5py_train_spects_144k.h5'
# frame
with h5py.File(h5py_path_frames, 'w') as hf:
    hf.create_dataset('train_frames', data=sample_frames)
# audio
with h5py.File(h5py_path_audios, 'w') as hf:
    hf.create_dataset('train_audios', data=sample_audios)
# spectrogram
with h5py.File(h5py_path_spects, 'w') as hf:
    hf.create_dataset('train_spects', data=sample_spects)


# # read h5 data
# with h5py.File(h5py_path_frames, 'r') as hf:
#     train_frames = hf['train_frames'][:]
