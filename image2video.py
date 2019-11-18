#coding:utf-8
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw,ImageFont

PATH_IMGS = 'zurich'
PATH_OUT_PATH = 'rename_images'

output_shape = (2048,1024)


fps = 30
#fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')


jpgNames = os.listdir(PATH_IMGS)

sliceIdx = 1


i = 0
while(i < len(jpgNames)):
    video_writer = cv2.VideoWriter(filename=jpgNames[i+19].split('.png')[0]+ '.avi', fourcc=fourcc, fps=fps, frameSize=output_shape)
    
    curImgPath = os.path.join(PATH_IMGS, jpgNames[i+17])
    img = cv2.imread(filename=curImgPath)
    cv2.waitKey(100)
    video_writer.write(img)
    
    curImgPath = os.path.join(PATH_IMGS, jpgNames[i+18])
    img = cv2.imread(filename=curImgPath)
    cv2.waitKey(100)
    video_writer.write(img)

    curImgPath = os.path.join(PATH_IMGS, jpgNames[i+19])
    img = cv2.imread(filename=curImgPath)
    cv2.waitKey(100)
    video_writer.write(img)

    print(jpgNames[i+19] + ' done!')
    i += 30


'''
for curImgName in jpgNames:
    curImgPath = os.path.join(PATH_IMGS, curImgName)

    if(i % fps == 0):
        video_writer = cv2.VideoWriter(filename=jpgNames[i+19].split('.png')[0]+ '.avi', fourcc=fourcc, fps=fps, frameSize=output_shape)

    img = cv2.imread(filename=curImgPath)
    img = cv2.resize(img, output_shape, interpolation = cv2.INTER_LINEAR)
    cv2.waitKey(100)
    video_writer.write(img)
    print(curImgName + ' done!')
    #cv2.imwrite(PATH_OUT_PATH + '/image'+str(i)+'.jpg', img)
    if(i % fps == fps - 1):
        video_writer.release()
        sliceIdx += 1
    i += 1
'''
