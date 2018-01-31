#!/usr/bin/python3
#-*- encoding: utf-8 -*-

import librosa
import librosa.display
import numpy as np
import playsound
import os
from util import Util
from PIL import Image
import cv2

def read_spectrogram_img(img_path):
    img = Image.open(img_path)
    return np.array(img)

def save_spectrogram_img(audio_path, img_path, bins):
    print(f'save_spectrogram_img {audio_path} -> {img_path}')
    s = calc_spectrogram(audio_path, bins=bins)
    if s is None:
        print(f'Failed to load:{audio_path}')
        return
    s_uint8 = (((s - s.min()) / (s.max() - s.min())) * 255.9).astype(np.uint8)
    Util.create_dir(os.path.dirname(img_path))
    Image.fromarray(s_uint8).save(img_path)

def calc_spectrogram(audio_path, window_ms=100, every_ms=50, bins=64, plot=False, play=False):
    try:
        y, sr = librosa.load(audio_path)
    except Exception as e:
        print(e)
        return None
    # applying xx ms window every yy ms
    n_fft = int(window_ms / 1000 * sr)
    hop_length = int(every_ms / 1000 * sr)
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    s = librosa.power_to_db(s, ref=np.max)
    s = cv2.resize(s, dsize=(s.shape[1], bins), interpolation=cv2.INTER_CUBIC)
    # print(s.shape, n_fft, hop_length)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(s, y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram %d x %d' % (s.shape[0], s.shape[1]))
        plt.tight_layout()
        plt.show()
    if play:
        playsound.playsound(audio_path)
    return s

def plot_spectrogram(audio_path, play=False):
    y, sr = librosa.load(audio_path)
    s = librosa.feature.melspectrogram(y=y, sr=sr)
    s = librosa.power_to_db(s, ref=np.max)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(s, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram %d x %d' % (s.shape[0], s.shape[1]))
    plt.tight_layout()
    plt.show()
    if play:
        playsound.playsound(audio_path)

if __name__ == '__main__':
    pass
    audio_path = 'data/train_audio/none/s0531rluyfo.sec105_5.mp3'
    image_path = 'data/train_image/none/s0531rluyfo.sec105_5.png'

    # plot_spectrogram(audio_path, play=False)
    calc_spectrogram(audio_path, plot=False, play=False)
    # calc_spectrogram(audio_path, window_ms=25, every_ms=10, plot=True, play=True)
    # save_spectrogram_img(audio_path, image_path)
    # img = read_spectrogram_img(image_path)
    # print('image:', img.shape, img.dtype)