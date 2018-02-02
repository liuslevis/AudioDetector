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
from pathlib import Path

def read_spectrogram_img(img_path):
    img = Image.open(img_path)
    return np.array(img)

def save_spectrogram_img(audio_path, img_path, window_ms, every_ms, bins=64, length=-1):
    print(f'save_spectrogram_img {audio_path} -> {img_path}')
    s = calc_spectrogram(audio_path, bins=bins)
    if s is None:
        print(f'Failed to load:{audio_path}')
        return
    s_uint8 = (((s - s.min()) / (s.max() - s.min())) * 255.9).astype(np.uint8)
    Util.create_dir(os.path.dirname(img_path))
    if length == -1:
        Image.fromarray(s_uint8).save(img_path)
    else:
        for i in range(int(s_uint8.shape[1] / length)):
            p = i * length
            q = (i + 1) * length
            image_part = s_uint8[:, p:q]
            if image_part.shape == (bins, length):
                path = Path(img_path).with_suffix(f'.part{i}.png')
                print(path)
                Image.fromarray(image_part).save(path)


def calc_spectrogram(audio_path, window_ms=25, every_ms=10, bins=64, plot=False, play=False):
    try:
        y, sr = librosa.load(audio_path)
    except Exception as e:
        print(e)
        return None
    # applying xx ms window every yy ms
    n_fft = int(window_ms * 1.0 / 1000 * sr)
    hop_length = int(every_ms * 1.0 / 1000 * sr)
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

    audio_path = 'data/train_audio/none/f0530m8fjco.p701.2.sec0_5.mp3'
    image_path = 'data/train_image/none/f0530m8fjco.p701.2.sec0_5.png'

    # plot_spectrogram(audio_path, play=False)
    # calc_spectrogram(audio_path, plot=False, play=False)
    # calc_spectrogram(audio_path, window_ms=25, every_ms=10, plot=True, play=True)
    save_spectrogram_img(audio_path, image_path, window_ms=25, every_ms=10, bins=64, length=101)
    # img = read_spectrogram_img(image_path)
    # print('image:', img.shape, img.dtype)