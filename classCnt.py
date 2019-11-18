from glob import glob
import cv2
import numpy as np 
pixel = [0.01152455, 0.06982842, 0.01862598, 0.64799141, 0.48393349,
                0.34624363, 2.03863404, 0.76879304, 0.02670268, 0.36683323,
                0.10595419, 0.34916275, 3.15175893, 0.06070994, 1.58812503,
                1.80554863, 1.82405714, 4.30866236, 1.02691057]

pixels = np.array([2036049361, 336031674, 1259776091, 36211223, 
            48487162, 67768934, 11509943, 30521298, 878734354, 
            63965201, 221459496, 67202363, 7444910, 386502819, 
            14775009, 12995807, 12863940, 5445904, 22849664])

label_colours_cityscapes = [[128, 64, 128], [244, 35, 231], [69, 69, 69]
                # 0 = road, 1 = sidewalk, 2 = building
                ,[102, 102, 156], [190, 153, 153], [153, 153, 153]
                # 3 = wall, 4 = fence, 5 = pole
                ,[250, 170, 29], [219, 219, 0], [106, 142, 35]
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,[152, 250, 152], [69, 129, 180], [219, 19, 60]
                # 9 = terrain, 10 = sky, 11 = person
                ,[255, 0, 0], [0, 0, 142], [0, 0, 69]
                # 12 = rider, 13 = car, 14 = truck
                ,[0, 60, 100], [0, 79, 100], [0, 0, 230]
                # 15 = bus, 16 = train, 17 = motocycle
                ,[119, 10, 32]]
                # 18 = bicycle

if __name__ == "__main__":
    imgs = glob('train_labels/*.png')

    num_classes = 19
    
    class_cnt = [0 for _ in range(num_classes)]


    for img_name in imgs:
        img = cv2.imread(img_name)[:,:,0]
        for cur_class in range(num_classes):
            #isIn = (cur_class in img)
            #class_cnt[cur_class] += isIn
            class_cnt[cur_class] += np.sum(img == cur_class)
        #print(isIn, 'wall', img_name )
    
    print(class_cnt)

        
