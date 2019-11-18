import os
import numpy as np
from PIL import Image
from glob import glob
from skimage.io import imread, imsave

#-------------------------SINGLE SCALE -------------------------------------

image_dirs = ['test/']
#label_dirs = ['images/train_labels']


valid_file_name = 'camvid_test_list.txt'


if __name__ == "__main__":
    f = open(valid_file_name, 'w')

    for i in range(len(image_dirs)):
        all_images = glob(image_dirs[i]+'*.png')
        for image_name in all_images:
            label_name = image_name.replace('test', 'test_labels')
            image_name = image_name.replace('_L','')
            f.write(image_name + ' ')
            f.write(label_name + '\n')
            print(image_name)


