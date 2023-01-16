from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('../')

import argparse
import cv2
import torch
from tkinter import filedialog
from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain

parser = argparse.ArgumentParser(description='TCTrack demo')
parser.add_argument('--config', type=str, default='../experiments/TCTrack/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='./snapshot/param.pth', help='model name')
args = parser.parse_args()

def get_frames(video_root):
    if video_root.endswith('avi') or video_root.endswith('mp4'):
        cap = cv2.VideoCapture(video_root)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break

def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # load video
    video_root = filedialog.askopenfilename(initialdir='/', title="choose your file",
                                            filetypes=(
                                            ("all files", "*.*"), ("avi files", "*.avi"), ("mp4 files", "*.mp4")))

    # create model
    model = ModelBuilder_tctrack('test')

    # load model
    model = load_pretrain(model, args.snapshot).eval().to(device)

    # build tracker
    tracker = TCTrackTracker(model)

    first_frame = True


    video_name = video_root.split('/')[-1].split('.')[0]


    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    index = 1
    for frame in get_frames(video_root):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame, index)
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 5)
            cv2.imshow(video_name, frame)
            if cv2.waitKey(1) == 27:
                break
            index = index + 1

if __name__ == '__main__':
    main()