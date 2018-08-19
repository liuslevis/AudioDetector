#!/usr/bin/python3
#-*- encoding: utf-8 -*-

import librosa
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

def save_feature_img(audio_path, img_path, window_ms, every_ms, bins, length=-1, mfcc=True):
    print(f'save_feature_img {audio_path} -> {img_path}')
    Util.create_dir(os.path.dirname(img_path))
    s = None
    if mfcc:
        s = calc_mfcc(audio_path, window_ms, every_ms, bins, plot=False)
        # s_uint8 = (((s - s.min()) / (s.max() - s.min())) * 255.9).astype(np.uint8)
        # s_uint8 = (((s - min_) / (max_ - min_)) * 255.9).astype(np.uint8)
        # print(np.percentile(s, q=0.9), s_uint8)
    else:
        s = calc_spectrogram(audio_path, window_ms, every_ms, bins, plot=False)
        s = (((s - s.min()) / (s.max() - s.min())) * 255.9).astype(np.uint8)

    if s is None:
        print(f'Failed to load:{audio_path}')
        return
    if length == -1:
        Image.fromarray(s).save(img_path)
    else:
        for i in range(int(s.shape[1] / length)):
            p = i * length
            q = (i + 1) * length
            image_part = s[:, p:q]
            if image_part.shape == (bins, length):
                if mfcc:
                    path = Path(img_path).with_suffix(f'.part{i}.npy')
                    image_part = image_part.reshape((*image_part.shape, 1))
                    np.save(path, image_part)
                    print(f'\t{path} {image_part.shape}')
                else:
                    path = Path(img_path).with_suffix(f'.part{i}.png')
                    Image.fromarray(image_part).save(path)
                    print(f'\t{path}')


def calc_mfcc(audio_path, window_ms, every_ms, bins, plot=False):
    try:
        y, sr = librosa.load(audio_path)
    except Exception as e:
        print(e)
        return None
    # applying xx ms window every yy ms
    n_fft = int(window_ms * 1.0 / 1000 * sr)
    hop_length = int(every_ms * 1.0 / 1000 * sr)
    s = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=bins, n_fft=n_fft, hop_length=hop_length)
    if plot:
        import matplotlib.pyplot as plt
        import librosa.display
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(s, x_axis='time')
        plt.colorbar()
        plt.title(f'MFCC {s.shape[0]} x {s.shape[1]}')
        plt.tight_layout()
        plt.show()
    return s

def calc_spectrogram(audio_path, window_ms=25, every_ms=10, bins=64, plot=False):
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
    if plot:
        import matplotlib.pyplot as plt
        import librosa.display
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(s, y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel spectrogram {s.shape[0]} x {s.shape[1]}')
        plt.tight_layout()
        plt.show()
    return s

def play_audio(audio_path):
    playsound.playsound(audio_path)

def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    s = librosa.feature.melspectrogram(y=y, sr=sr)
    s = librosa.power_to_db(s, ref=np.max)
    s = librosa.feature.mfcc(S=s)
    import matplotlib.pyplot as plt
    import librosa.display
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(s, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram %d x %d' % (s.shape[0], s.shape[1]))
    plt.tight_layout()
    plt.show()

def plot_mfcc(audio_path):
    y, sr = librosa.load(audio_path)
    s = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    print(s.shape)
    import matplotlib.pyplot as plt
    import librosa.display
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(s, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram %d x %d' % (s.shape[0], s.shape[1]))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    image_path = 'data/test.png'
    audio_path = 'data/train_audio/laugh/d0532528se9.p712.1.sec32_4.mp3'

    # play_audio(audio_path)
    # plot_spectrogram(audio_path)
    # plot_mfcc(audio_path)
    # calc_mfcc(audio_path, window_ms=100, every_ms=50, bins=64, plot=True)
    # calc_spectrogram(audio_path, plot=False)
    # calc_spectrogram(audio_path, window_ms=25, every_ms=10, plot=True)
    save_feature_img(audio_path, image_path, window_ms=50, every_ms=10, bins=64, length=100)
    # img = read_spectrogram_img(image_path)
    # print('image:', img.shape, img.dtype)