from email.mime import base
import numpy as np
import cv2
import os
import random
from random import sample
random.seed(100)
GID_map = {"建筑": [255,0,0],
           "田地": [0,255,0],
           "森林": [0,255,255],
           "草地": [255,255,0],
           "水体": [0,0,255]
}
FU_map = {
    "建筑": [219, 95, 87],
    "基础设施": [219, 151, 87],
    "工矿用地": [219, 208, 87], 
    "城市绿地": [173, 219, 87], 
    "耕地": [117, 219, 87],
    "园地": [123, 196, 123],
    "牧场": [88, 177, 88],
    "森林": [0, 128, 0],
    "灌木": [88, 176, 167],
    "裸地": [153, 93, 19],
    "湿地": [87, 155, 219],
    "水体": [0, 98, 255]
}
                                

def translabel(map, label):
    h, w = label.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for i, index in enumerate(map):
        rgb_image[label == i] = map[index]

    return rgb_image

if __name__ == "__main__":
    base_dir = ""
    list_path = ""

    if not os.path.exists(os.path.join(base_dir, "rgb_label_sample")):
            os.makedirs(os.path.join(base_dir, "rgb_label_sample"))
    with open(list_path, "rb") as f:
            datalist = f.readlines()

    label_list = [
                k[1]
                for k in map(
                    lambda x: x.decode("utf-8").strip("\n").strip("\r").split("\t"), datalist
                )
            ]
    sample_list = label_list
    for label_path in sample_list:
        label = cv2.imread(os.path.join(base_dir, label_path), cv2.IMREAD_GRAYSCALE)
        image = translabel(GID_map, label)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filename = label_path.split('/')[-1]
        cv2.imwrite(os.path.join(base_dir, "rgb_label_sample", filename), image)
    
