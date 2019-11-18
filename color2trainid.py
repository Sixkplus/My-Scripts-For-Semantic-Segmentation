import cv2
import numpy as np
from glob import glob

all_name = glob('train/*.png')

color2id = {(128, 0, 0):0, (128,128,0):1, (128,128,128):2, (64,0,128):3, (192,128,128):4, (128,64,128):5, (64,64,0):6, (64,64,128):7, (192,192,128):8, (0,0,192):9, (0,128,192):10}


i = 0
for cur_name in all_name:
    cur_img = cv2.imread(cur_name)

    cur_img = cv2.cvtColor(np.uint8(cur_img), cv2.COLOR_BGR2RGB)

    h, w, _ = cur_img.shape
    trainId_img = np.zeros([h,w], dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if tuple(cur_img[i,j]) in color2id:
                trainId_img[i,j] = color2id[tuple(cur_img[i,j])]
            else:
                trainId_img[i,j] = 255
    
    #trainId_img = color2id[cur_img.astype(int)]
    cv2.imwrite(cur_name.replace('train', 'train_labels'), trainId_img)
    print(i, cur_name.replace('train', 'train_labels'))
    i+=1








'''
Used classes
----------------------------------
128 0 0		Building
128 128 0	Tree
128 128 128	Sky
64 0 128	Car
192 128 128	SignSymbol
128 64 128	Road
64 64 0		Pedestrian
64 64 128	Fence
192 192 128	Column_Pole
0 0 192		Sidewalk
0 128 192	Bicyclist
others      void


'''



'''
Others
-----------------------------
64 128 64	Animal
192 0 128	Archway

0 128 64	Bridge


64 0 192	CartLuggagePram
192 128 64	Child


128 0 192	LaneMkgsDriv
192 0 64	LaneMkgsNonDriv
128 128 64	Misc_Text
192 0 192	MotorcycleScooter
128 64 64	OtherMoving
64 192 128	ParkingBlock


128 128 192	RoadShoulder



64 128 192	SUVPickupTruck
0 0 64		TrafficCone
0 64 64		TrafficLight
192 64 128	Train

192 128 192	Truck_Bus
64 0 64		Tunnel
192 192 0	VegetationMisc
0 0 0		Void
64 192 0	Wall
'''