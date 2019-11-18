from glob import glob
import cv2
import numpy as np
import tensorflow as tf
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

label_colours_freetech = [[153,153,153], [128, 64, 128], [0, 0, 142]
                # 0 = background, 1 = road, 2 = vehicle
                ,[255, 0, 0], [219, 19, 60], [219, 219, 0]]
                # 3 = rider, 4 = walker, 5 = cone

label_colours_agric = [[153,153,153], [255,0,0], [219, 219, 0]
                # 0 = background, 1 = 烤烟, 2 = 玉米
                ,[0, 0, 255]]
                # 3 = 薏米仁

id_list_cityscapes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

trainid_map = np.array([0,0,0,0,0,0,0,0,1,0,0,2,3,4,0,0,0,5,0,6,7,8,9,10,11,12,13,14,15,0,0,16,17,18])
id_list_freetech = [0,1,2,5,4,7]
id_list_agric = [0,1,2,3]

def decode_labels(mask, img_shape, num_classes):
    if num_classes == 6:
        color_table = label_colours_freetech
    elif num_classes == 4:
        color_table = label_colours_agric
    else:
        color_table = label_colours_cityscapes

    color_mat = tf.constant(color_table, dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, int(img_shape[0]), int(img_shape[1]), 3))
    #pred = tf.reshape(pred, (1, 1024, 2048, 3))
    #pred = tf.reshape(pred, (1, 256, 256, 3))
    
    return pred

PATH_FOLDERS = './cityscapes/'
folders = ['results_bl2_ffw12_628', 'results_bl2_rgc12_668', 'results_bl2_rgfs_rgc12_691']


img_placeholder = tf.placeholder(dtype = tf.uint8, shape = [1024, 2048])

num_classes = 19

output_color = decode_labels(img_placeholder, [1024, 2048], num_classes)


if __name__ == "__main__":
    sess = tf.Session()

    for folder in folders:
        cur_folder_path = PATH_FOLDERS + folder + '/*id.png'
        imgs = glob(cur_folder_path)


        for img_name in imgs:
            img = cv2.imread(img_name)[:,:,0]

            img = trainid_map[img]

            output = sess.run(output_color, feed_dict={img_placeholder:img}).reshape([1024, 2048, 3])

            output = cv2.cvtColor(np.uint8(output), cv2.COLOR_RGB2BGR)

            cv2.imwrite(img_name.replace('id','color'), output)

        

        
