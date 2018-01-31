#!/usr/bin/python3
#-*- encode:utf-8 -*-

import cv2
import sys, os, time
import skvideo.io
import subprocess
import pathlib
import hashlib
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from threading import Thread
from queue import Queue
import datetime
import io
import numpy as np
import platform

if 'centos' in platform.dist():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class Util(object):
    def sec2str(sec):
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def get_time_str():
        return datetime.datetime.now().strftime("%Y%m%d%H%M")

    def run_shell_cmd(cmd):
        subprocess.call(cmd, shell=True)

    def run_shell_path(path):
        subprocess.call('chmod +x ' + path, shell=True)
        subprocess.call(path, shell=True)

    def create_dir(directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    def is_centos():
        return 'centos' in platform.dist()

    def is_ubuntu():
        return 'Ubuntu' in platform.dist()
        
    def is_mac():
        return sys.platform == 'darwin'

    def is_win():
        return sys.platform == 'win32'

    def is_linux():
        return sys.platform == 'linux'

    def use_cv2_video_func():
        return Util.is_mac() or Util.is_win()
        # return False # tlinux's cv2.VideoCapture dont work
        
    def get_fps(video_path):
        ret = None
        if Util.use_cv2_video_func():
            cap = cv2.VideoCapture(video_path)
            ret = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
        else:
            info = skvideo.io.ffprobe(video_path)
            if 'video' in info and '@r_frame_rate' in info['video']:
                ret = int(eval(info['video']['@r_frame_rate']))
            else:
                ret = 0
        return ret

    def get_video_sec(video_path):
        result = subprocess.Popen(["ffprobe", video_path],
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        for line in result.stdout.readlines():
            s = str(line)
            if 'Duration' in s:
                ts = s.split(',')[0].split(': ')[1]
                print(ts)
                pt = datetime.datetime.strptime(ts, '%H:%M:%S.%f')
                n_sec = pt.second + pt.minute * 60 + pt.hour * 3600
                return n_sec
        return 0

    def loop_image_dir(directory, looper, fps, per_sec):
        '''
        @return n_frame
        '''
        i_frame = 0
        for filename in os.listdir(directory):
            if filename.endswith(filename):
                path = os.path.join(directory, filename)
                img = cv2.imread(path)
                i_sec = int(i_frame / fps)
                if looper: looper(img, i_frame, i_sec)
                i_frame += 1 * per_sec * fps
        return i_frame

    def get_i_sec_frame(video_path, i_sec):
        stream = cv2.VideoCapture(video_path) if Util.use_cv2_video_func() else skvideo.io.vreader(video_path)
        fps = Util.get_fps(video_path)
        if fps == 0: 
            return None
        i = 0
        while True:
            if Util.use_cv2_video_func():
                grabbed, frame = stream.read()
                if not grabbed: 
                    break
                if i / fps == i_sec:
                    return frame
                    stream.release()
                i += 1
            else:
                frame = next(stream, None)
                if frame is None:
                    break
                frame = frame[...,::-1] # RGB -> BGR
                if i / fps == i_sec:
                    return frame
                i += 1
        if Util.use_cv2_video_func():
            stream.release()
        return None

    # not fast thought
    def fast_loop_frame(video_path, looper, per_sec, max_sec):
        '''
        @return n_frame
        '''
        i_frame = 0
        fps = Util.get_fps(video_path)
        fvs = FileVideoStream(video_path, queueSize=4096).start()
        time.sleep(5.0)
        while fvs.more():
            time.sleep(0.001)
            frame = fvs.read()
            if i_frame % fps == 0 and (i_frame / fps) % per_sec == 0:
                i_sec = int(i_frame / fps)
                if looper: looper(frame, i_frame, i_sec)
            if (i_frame / fps) > max_sec:
                break
            i_frame += 1
        return i_frame

    def loop_frame(video_path, looper, per_sec, max_sec):
        '''
        @return n_frame
        '''
        i_frame = 0
        fps = Util.get_fps(video_path)
        stream = cv2.VideoCapture(video_path) if Util.use_cv2_video_func() else skvideo.io.vreader(video_path)
        while True:
            frame = None
            if Util.use_cv2_video_func():
                grabbed, frame = stream.read()
                if not grabbed:
                    break
            else:
                frame = next(stream, None)
                if frame is None:
                    break
                frame = frame[...,::-1] # RGB -> BGR
            if i_frame % fps == 0 and (i_frame / fps) % per_sec == 0:
                i_sec = int(i_frame / fps)
                if looper: looper(frame, i_frame, i_sec)
            if (i_frame / fps) > max_sec:
                break
            i_frame += 1

        if Util.use_cv2_video_func(): 
            stream.release()
        return i_frame

    def plot_keras_history(history, plot_path, log_path, model):
        log_path = log_path.replace('detail.txt', 'acc{0:.2f}.txt'.format(history.history['acc'][-1]))
        with open(log_path, 'w') as f:
            for key in ['val_acc', 'acc', 'val_loss', 'loss']:
                try:
                    f.write('\n{}='.format(key))
                    nums = history.history[key]
                    f.write(','.join(list(map(lambda x:'%.4f' % x, nums))))
                except Exception as e:
                    pass
        if Util.is_linux():
            return
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.add_subplot(2,2,1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        axes = plt.gca()
        axes.set_ylim([0, 1])

        # summarize history for loss
        fig.add_subplot(2,2,2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.savefig(plot_path)

    def plot_image(image):
        if Util.is_linux():
            return
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.show()

    def plot_dbs(plot_path, dbs, db_max):
        if Util.is_linux():
            return
        import matplotlib
        import matplotlib.pyplot as plt

        n = len(dbs)
        fig, ax = plt.subplots(figsize=(n / 60 * 5, 2.5))
        ax.set_xticks(np.arange(0, n, 10))
        formatter = matplotlib.ticker.FuncFormatter(lambda sec, x: time.strftime('%M:%S', time.gmtime(sec)))
        ax.xaxis.set_major_formatter(formatter)
        x1 = np.arange(0, n)
        y1 = dbs
        x2 = np.arange(0, n)
        y2 = db_max * np.ones((n))
        ax.plot(x1, y1, 'b-', x2, y2, 'r-')
        ax.set_ylabel('dB')
        ax.set_xlabel('sec')
        ax.set_ylim((-120, 0))
        ax.set_xlim((0, n))
        fig.savefig(plot_path)

    def rect_smoba_killinfo(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 70 / 480)
        y2 = int(w * 120 / 480)
        x1 = int(h * 300 / 848)
        x2 = int(h * 548 / 848)
        return x1, y1, x2, y2

    def rect_smoba_skill_1(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 385 / 480)
        y2 = int(w * 455 / 480)
        x1 = int(h * 600 / 848)
        x2 = int(h * 670 / 848)
        return x1, y1, x2, y2

    def rect_smoba_middle_hero(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 190 / 480)
        y2 = int(w * 300 / 480)
        x1 = int(h * 390 / 848)
        x2 = int(h * 460 / 848)
        return x1, y1, x2, y2

    def rect_smoba_grid_hero(image, i, j):
        x1, y1, x2, y2 = Util.rect_middle_hero(image)
        dx = i * (x2 - x1)
        dy = j * (y2 - y1)
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy

    def rect_pubg_killinfo(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 300 / 450)
        y2 = int(w * 340 / 450)
        x1 = int(h * 300 / 800) 
        x2 = int(h * 500 / 800)
        return x1, y1, x2, y2

    def rect_pubg_gun(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 380 / 450)
        y2 = int(w * 420 / 450)
        x1 = int(h * 360 / 800)
        x2 = int(h * 440 / 800)
        return x1, y1, x2, y2

    def rect_pubg_win(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 0 / 450)
        y2 = int(w * 100 / 450)
        x1 = int(h * 600 / 800) 
        x2 = int(h * 800 / 800)
        return x1, y1, x2, y2
        
    def rect_pubg_team(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 350 / 450)
        y2 = int(w * 450 / 450)
        x1 = int(h * 0 / 800) 
        x2 = int(h * 100 / 800)
        return x1, y1, x2, y2

    def rect_pubg_screen(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = 0
        y2 = int(w)
        x1 = 0
        x2 = int(h)
        return x1, y1, x2, y2

    def rect_speedm_drift_button(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 225 / 340)
        y2 = int(w * 305 / 340)
        x1 = int(h * 490 / 605) 
        x2 = int(h * 570 / 605)
        return x1, y1, x2, y2

    def rect_speedm_drift_info(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 260 / 340)
        y2 = int(w * 290 / 340)
        x1 = int(h * 310 / 605) 
        x2 = int(h * 430 / 605)
        return x1, y1, x2, y2

    def rect_speedm_record(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 20 / 340)
        y2 = int(w * 100 / 340)
        x1 = int(h * 230 / 605) 
        x2 = int(h * 380 / 605)
        return x1, y1, x2, y2

    def rect_speedm_win(image):
        w = image.shape[0]
        h = image.shape[1]
        y1 = int(w * 70 / 340)
        y2 = int(w * 190 / 340)
        x1 = int(h * 330 / 605) 
        x2 = int(h * 605 / 605)
        return x1, y1, x2, y2

    # height, width, channel
    def input_shape_speedm_drift_button():
        return (80, 80, 3)

    def input_shape_speedm_drift_info():
        return (30, 120, 3)

    def input_shape_speedm_record():
        return (80, 150, 3)

    def input_shape_speedm_win():
        return (120, 275, 3)

    def input_shape_pubg_killinfo():
        return (40, 200, 3)

    def input_shape_pubg_gun():
        return (50, 100, 3)
    
    def input_shape_pubg_win():
        return (50, 100, 3)
    
    def input_shape_pubg_team():
        return (100, 100, 3)
    
    def input_shape_pubg_screen():
        return (45, 80, 3)

    def input_shape_smoba_skill_1():
        return (50, 50, 3)

    def input_shape_smoba_hero():
        return (90, 70, 3)

    def crop_and_resize(image, rect_func, size):
        x1, y1, x2, y2 = rect_func(image)
        image = image[y1:y2, x1:x2]
        return cv2.resize(image, size)

    def crop_smoba_skill_1(image):
        return Util.crop_and_resize(image, Util.rect_skill_1, size=(50, 50))

    def crop_smoba_middle_hero(image):
        return Util.crop_and_resize(image, Util.rect_middle_hero, size=(70, 90))

    def crop_smoba_grid_hero(image, i, j):
        return Util.crop_and_resize(image, Util.rect_grid_hero, size=(70, 90))

    def crop_pubg_killinfo(image):
        return Util.crop_and_resize(image, Util.rect_pubg_killinfo, size=(200, 40))

    def crop_pubg_gun(image):
        return Util.crop_and_resize(image, Util.rect_pubg_gun, size=(200, 100))

    def crop_pubg_win(image):
        return Util.crop_and_resize(image, Util.rect_pubg_win, size=(100, 50))

    def crop_pubg_team(image):
        return Util.crop_and_resize(image, Util.rect_pubg_team, size=(100, 100))

    def crop_pubg_screen(image):
        return Util.crop_and_resize(image, Util.rect_pubg_screen, size=(160, 90))

    def crop_speedm_drift_button(image):
        return Util.crop_and_resize(image, Util.rect_speedm_drift_button, size=(80, 80))

    def crop_speedm_drift_info(image):
        return Util.crop_and_resize(image, Util.rect_speedm_drift_info, size=(120, 30))

    def relpath(file, path):
        return os.path.join(os.path.dirname(file), path)

    def get_file_paths(directory, suffixes):
        paths = []
        for filename in os.listdir(directory):
            for suffix in suffixes:
                if filename.endswith(suffix):
                    paths.append(os.path.join(directory, filename))
        return paths

    def create_watermark(text, path, video_path):
        # 灰底黑字
        font_size = 40
        pad_top = pad_bottom = 5
        pad_left = pad_right = 10
        font = ImageFont.truetype("./resource/font/SourceHanSansCN-Bold.otf", font_size)
        w0, h0 =  (600, font_size * 3)
        img = Image.new("RGBA", (w0, h0), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        textsize = draw.multiline_textsize(text, font)
        rect_x0 = 0
        rect_y0 = 0
        rect_x1 = pad_left + pad_right + textsize[0]
        rect_y1 = pad_top + pad_bottom + textsize[1]
        draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(0,0,0,140), outline=None)
        draw.text((pad_left, pad_top), text, fill=(255,255,255,255), font=font)

        # rescale
        frame = Util.get_i_sec_frame(video_path, i_sec=1)
        if frame is not None:
            w1, h1, _ = frame.shape
            if 0 < h1 < 1920:
                scale = h1 / 1920
                img.thumbnail((int(w0 * scale), int(h0 * scale)), Image.ANTIALIAS)
        img.save(path)
        
        # 白字
        # font_size = 40
        # font = ImageFont.truetype("./resource/font/SourceHanSansCN-Bold.otf", font_size)
        # img = Image.new("RGBA", (600, font_size * 3), (0, 0, 0, 0))
        # draw = ImageDraw.Draw(img)
        # draw.text((0, 0), text, (221,221,221), font=font)
        # draw = ImageDraw.Draw(img)
        # img.save(path)
        #
        # 灰底深灰字
        # font_size = 40
        # font = ImageFont.truetype("./resource/font/SourceHanSansCN-Bold.otf", font_size)
        # img = Image.new("RGBA", (600, font_size * 3), (0, 0, 0, 0))
        # draw = ImageDraw.Draw(img)
        # textsize = draw.multiline_textsize(text, font)
        # draw.rectangle([(0, 0), textsize], fill=(180,180,180,150), outline=None)
        # draw.text((0, 0), text, fill=(80,80,80,200), font=font)
        # img.save(path)

    def get_md5(text_li):
        h = hashlib.md5()
        for text in text_li:
            h.update(text.encode('utf-8'))
        return h.hexdigest()

class FileVideoStream:
    def __init__(self, path, queueSize):
        self.stream = cv2.VideoCapture(path) if Util.use_cv2_video_func() else skvideo.io.vreader(path)
        self.stopped = False
        self.queue = Queue(maxsize=queueSize)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            if self.stopped:
                return

            if not self.queue.full():
                frame = None
                if Util.use_cv2_video_func():
                    grabbed, frame = self.stream.read()
                    if not grabbed:
                        self.stop()
                        return
                else:
                    frame = next(self.stream, None)
                    if frame is None:
                        self.stop()
                    frame = frame[...,::-1] # RGB -> BGR
                self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def more(self):
        return self.queue.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        if Util.use_cv2_video_func():
            self.stream.release()

