import os
import datetime
import time
from util import Util
from pathlib import Path
from util_audio import *
from datetime import datetime

VIDEO_SUFFIX = '.mp4'
AUDIO_SUFFIX = '.mp3'
IMAGE_SUFFIX = '.png'

def timestr_sec(timestr):
    pt = datetime.strptime(timestr, '%M:%S')
    total_seconds = pt.second + pt.minute * 60 + pt.hour * 3600
    return total_seconds

def read_txt(path):
    ret = []
    with open(path) as f:
        last_end = 0
        for line in f.readlines():
            parts = line.lstrip().rstrip().split(',')
            if len(parts) == 2:
                begin, end = parts
                begin_sec, end_sec = timestr_sec(begin), timestr_sec(end)
                if begin_sec % 2 == 1: 
                    begin_sec -= 1
                if end_sec % 2 == 1:
                    end_sec += 1
                if last_end < begin_sec:
                    ret.append((last_end, begin_sec, 'none'))
                ret.append((begin_sec, end_sec, 'laugh'))
                last_end = end_sec
    return ret

def cut_audio(src: str, outdir: str, ss, t: int):
    if type(ss) == str:
        pt = datetime.datetime.strptime(ss, '%H:%M:%S')
        ss = pt.second + pt.minute * 60 + pt.hour * 3600
    input_audio = src.with_suffix(AUDIO_SUFFIX)
    output_audio = outdir / os.path.basename(src.with_suffix(f'.sec{ss}_{t}{AUDIO_SUFFIX}'))
    if not input_audio.exists():
        Util.run_shell_cmd(f'ffmpeg -i {src} {input_audio}')
    if not output_audio.exists():
        cmd = f'ffmpeg -ss {ss} -i {input_audio} -t {t} {output_audio}'
        print(cmd)
        Util.run_shell_cmd(cmd)
           
def gen_train_audio(raw_dir, todo_dir):
    print('gen_train_audio', raw_dir, '->', todo_dir)
    Util.create_dir(todo_dir)
    for root, dirs, filenames in os.walk(raw_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                video_path = (Path(root) / filename).with_suffix(VIDEO_SUFFIX)
                if not video_path.exists():
                    print('Error: path not exists', video_path)
                    continue
                for begin, end, label in read_txt(video_path.with_suffix('.txt')):
                    audio_dir = Path(todo_dir) / label
                    audio_name = video_path.with_suffix(f'.sec{begin}_{end - begin}{AUDIO_SUFFIX}').name
                    cmd = f'ffmpeg -y -ss {begin} -i {video_path} -t {end - begin} {audio_dir / audio_name}'
                    print(cmd)
                    Util.create_dir(audio_dir)
                    Util.run_shell_cmd(cmd)

def gen_train_img(train_audio_dir, train_image_dir, window_ms, every_ms, bins, length, mfcc):
    for root, dirs, filenames in os.walk(train_audio_dir):
        if len(filenames) > 1:
            for filename in filenames:
                if AUDIO_SUFFIX in filename:
                    image_path = (Path(root.replace(str(train_audio_dir), str(train_image_dir))) / filename).with_suffix(IMAGE_SUFFIX)
                    audio_path = Path(root) / filename
                    save_feature_img(audio_path, image_path, window_ms, every_ms, bins, length, mfcc)

if __name__ == '__main__':
    gen_train_audio(Path('data/raw_other'), Path('data/train_audio'))
    # 手工分类
    gen_train_img(Path('data/train_audio'), Path('data/train_image_win100ms'), window_ms=100, every_ms=50, bins=64, length=101, mfcc=False)
    gen_train_img(Path('data/train_audio'), Path('data/train_image_mfcc'), window_ms=25, every_ms=10, bins=64, length=100, mfcc=True)