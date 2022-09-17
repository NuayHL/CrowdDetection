import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
from pycocotools.coco import COCO

def coco_gt_to_txt(jsonfile, output_folder_name):
    coco = COCO(jsonfile)
    assert not os.path.exists(output_folder_name),'output folder already exists!'
    os.makedirs(output_folder_name)
    file_path = output_folder_name
    img_ids = coco.getImgIds()
    cates = coco.cats
    for imgid in img_ids:
        anns = coco.getAnnIds(imgid)
        with open(os.path.join(file_path, imgid+'.txt')) as f:
            for ann in anns:
                class_name = cates[ann['category_id']]['name']
                x1, y1 = ann['bbox'][:2]
                x2, y2 = ann['bbox'][:2] + ann['bbox'][2:]
                print(class_name,x1, y1, x2, y2, file=f)

if __name__ == '__main__':
    coco_gt_to_txt('CrowdHuman/annotation_val_coco_style.json','coco_val')