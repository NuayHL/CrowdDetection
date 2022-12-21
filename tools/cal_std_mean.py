import json
import cv2
import numpy as np
import os

path = os.getcwd()
os.chdir(os.path.join(path, '..'))

'''
format in BGR! Need to change to RGB
'''

def cal_img_mean_std(coco_style_annotation_path, image_path):
    with open(coco_style_annotation_path) as f:
        annotation = json.load(f)
    assert "images" in annotation.keys(),"Unknown format of annotation. The annotation must have key:\'images\'"

    mean = np.zeros((3)).astype(np.float_)
    std = np.zeros((3)).astype(np.float_)
    imgnum = len(annotation["images"])
    num_pixels = 0
    for idx, imgs in enumerate(annotation["images"]):
        real_img = cv2.imread(os.path.join(image_path,imgs["file_name"])).astype(np.float_)
        num_pixels += imgs["width"] * imgs["height"]
        for i in range(3):
            mean[i] += real_img[:, :, i].sum()
            std[i] += np.power(real_img[:, :, i],2).sum()
        progressbar(float((idx+1.0)/imgnum))

    mean = mean/(255*num_pixels)
    std = np.power(std/(255*255*num_pixels)-np.power(mean,2),0.5)

    return mean,std

def progressbar(percentage, endstr='', barlenth=20):
    if int(percentage)==1: endstr +='\n'
    print('\r[' + '>' * int(percentage * barlenth) +
          '-' * (barlenth - int(percentage * barlenth)) + ']',
          format(percentage * 100, '.1f'), '%', end=' '+endstr)

if __name__ == "__main__":
    print(os.getcwd())

    mean, std = cal_img_mean_std("ECP/day/labels/train_coco_style.json", "ECP/day/img/train")
    print("mean:", mean)
    #[0.4223358  0.44211456 0.46431773]
    print("std:", std)
    #[0.29363019 0.28503336 0.29044453]

    # ECP
    # mean: [0.35733526 0.36444973 0.34919569]
    # std: [0.23302755 0.23708822 0.23204576]