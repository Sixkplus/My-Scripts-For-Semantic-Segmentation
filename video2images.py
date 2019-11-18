from glob import glob
import cv2
import numpy as np
import os

video_path = 'Seq05VD.MXF'

image_save_path = video_path.split('.')[0]
if not os.path.exists(image_save_path):
    os.mkdir(image_save_path)

video = cv2.VideoCapture(video_path)

if video.isOpened():
    rval, frame = video.read()
    frame_index = -1
else:
    rval = False

while rval:
    cv2.imwrite(os.path.join(image_save_path, video_path.split('.')[0] + '_f' +"{:0>5d}".format(frame_index)) + ".png", frame)
    rval, frame = video.read()
    frame_index += 1


        

        
