import os
import datetime
import time
from util import Util
from pathlib import Path
from util_audio import *

raw_dir = Path('./data/raw')
todo_dir = Path('./data/todo') # mp3 切片，人工分类 
train_audio_dir = Path('./data/train_audio')
train_image_dir = Path('./data/train_image')

INPUT_SUFFIX = '.mp4'
AUDIO_SUFFIX = '.mp3'
IMAGE_SUFFIX = '.png'

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
    for root, dirs, filenames in os.walk(raw_dir):
        for filename in filenames:
            if filename.endswith(INPUT_SUFFIX):
                n_sec = Util.get_video_sec(str(raw_dir / filename))
                for i_sec in range(0, n_sec, 5):
                    cut_audio(raw_dir / filename, todo_dir, i_sec, 5)  
                # cut_audio(raw_dir / filename, todo_dir / 'none', '00:00:00', 5)
            print(raw_dir / filename)

def gen_train_img(train_audio_dir, train_image_dir, bins):
    for root, dirs, filenames in os.walk(train_audio_dir):
        if len(filenames) > 5:
            for filename in filenames:
                if AUDIO_SUFFIX in filename:
                    image_path = (Path(root.replace(str(train_audio_dir), str(train_image_dir))) / filename).with_suffix(IMAGE_SUFFIX)
                    audio_path = Path(root) / filename
                    if not image_path.exists():
                        save_spectrogram_img(audio_path, image_path, bins)                

if __name__ == '__main__':
    # gen_train_audio(raw_todo_dir, todo_dir)
    gen_train_img(train_audio_dir, train_image_dir, bins=64)