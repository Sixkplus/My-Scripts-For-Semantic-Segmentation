cript aims to obtain the compressed information
# Visualization the motion vectors and residual graph

import numpy as np 
import cv2
import os, glob
import time
from skimage.io import imread, imsave

from coviar import load
from coviar import get_num_frames

GOP_FRAMES_NUM = 12
PATH_RGB_FRAMES = 'rgb'

# The continuous MV and cummulative motion vectors
PATH_MV_CONT = 'mv_cont/'

# The residual graph
PATH_RES_CONT = 'res_cont/'

if not os.path.exists(PATH_MV_CONT):
    os.mkdir(PATH_MV_CONT)
if not os.path.exists(PATH_RES_CONT):
    os.mkdir(PATH_RES_CONT)

video_names = glob.glob('./video/*.avi')
video_names.sort()


def main():
    #for video_name in video_names[:2]:
    for video_name in video_names:
        fold_path = video_name.split('.avi')[0].split('/')[-1]
        path_mv = os.path.join(fold_path, PATH_MV_CONT)
        path_res = os.path.join(fold_path, PATH_RES_CONT)
        if not os.path.exists(path_mv):
            os.makedirs(path_mv)
        if not os.path.exists(path_res):
            os.makedirs(path_res)
        NUM_FRAMES = get_num_frames(video_name)
        print(NUM_FRAMES)
        # The index of GOP
        curGopIdx = 0
        for curGopIdx in range(max(NUM_FRAMES // GOP_FRAMES_NUM, 1)):
            for innerGopIdx in range(GOP_FRAMES_NUM):
                curFrameIdx = curGopIdx * GOP_FRAMES_NUM + innerGopIdx
                #rgbFrame = load(video_name, curGopIdx, innerGopIdx, 0, True)

                #start = time.time()
                print(video_name, curGopIdx, innerGopIdx)
                mvCont_origin = load(video_name, curGopIdx, innerGopIdx, 1, False)
                resCont = load(video_name, curGopIdx, innerGopIdx, 2, False)

                if mvCont_origin is None:
                    mvCont_origin = np.zeros([720,960,2], dtype=np.uint8)
                
                mvCont = mvCont_origin + 2048
                # (high_h, low_h, high_w, low_w)
                mvPng = np.array([((mvCont[:,:,0] >> 8) & 0xff) , (mvCont[:,:,0] & 0xff), ((mvCont[:,:,1] >> 8) & 0xff), (mvCont[:,:,1] & 0xff)], dtype = np.uint8)
                mvPng = np.transpose(mvPng, [1,2,0])

                

                imsave(path_mv+'/frame'+str(curFrameIdx)+'.png', mvPng)
                #save_mvPng = imread(path_mv+'/frame'+str(curFrameIdx)+'.png').astype(np.int16)

                #reload_mvCont = np.array([ (save_mvPng[:,:,0] << 8) + (save_mvPng[:,:,1]), (save_mvPng[:,:,2] << 8) + (save_mvPng[:,:,3]) ])
                #reload_mvCont = np.transpose(reload_mvCont, [1,2,0])
                #reload_mvCont -= 2048

                #print((reload_mvCont == mvCont_origin).min())
                if resCont is None:
                    resCont = np.zeros([720,960,3], dtype=np.uint8)
                
                resCont = np.round((resCont + 256)/2).astype(np.uint8)
                #resCont = np.abs(resCont)
                imsave(path_res+'/frame'+str(curFrameIdx)+'.png', resCont)
                cv2.imwrite(PATH_RES_CONT+fold_path+'.png', resCont)
                

if __name__ == "__main__":
    main()
