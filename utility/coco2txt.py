import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
import json
from pycocotools.coco import COCO

def progressbar(percentage, endstr='', barlenth=20):
    if int(percentage)==1: endstr +='\n'
    print('\r[' + '>' * int(percentage * barlenth) +
          '-' * (barlenth - int(percentage * barlenth)) + ']',
          format(percentage * 100, '.1f'), '%', end=' '+endstr)

def coco_gt_to_txt(jsonfile, output_folder_name):
    coco = COCO(jsonfile)
    assert not os.path.exists(output_folder_name),'output folder already exists!'
    os.makedirs(output_folder_name)
    file_path = output_folder_name
    img_ids = coco.getImgIds()
    total_imgs = float(len(img_ids))
    cates = coco.cats
    for i, imgid in enumerate(img_ids):
        ann_ids = coco.getAnnIds([imgid])
        with open(os.path.join(file_path, imgid+'.txt'), 'w') as f:
            for ann_id in ann_ids:
                ann = coco.anns[ann_id]
                class_name = cates[ann['category_id']]['name']
                x1, y1 = ann['bbox'][:2]
                x2 = ann['bbox'][0] + ann['bbox'][2]
                y2 = ann['bbox'][1] + ann['bbox'][3]
                print(class_name,x1, y1, x2, y2, file=f)
        progressbar((i+1)/total_imgs, barlenth=40)

def coco_dt_to_txt(gt_file, dt_file, output_folder_name):
    gt = COCO(gt_file)
    cates = gt.cats
    with open(dt_file,'r') as f:
        dt = json.load(f)
    assert not os.path.exists(output_folder_name), 'output folder already exists!'
    os.makedirs(output_folder_name)
    file_path = output_folder_name
    total_bboxes = float(len(dt))
    for i, dt_bbox in enumerate(dt):
        with open(os.path.join(file_path, dt_bbox['image_id']+'.txt'),'a') as f:
            class_name = cates[dt_bbox['category_id']]['name']
            score = dt_bbox['score']
            x1, y1 = dt_bbox['bbox'][:2]
            x2 = dt_bbox['bbox'][0] + dt_bbox['bbox'][2]
            y2 = dt_bbox['bbox'][1] + dt_bbox['bbox'][3]
            print(class_name,score,x1,y1,x2,y2,file=f)

        progressbar((i + 1) / total_bboxes, barlenth=40)


if __name__ == '__main__':
    coco_dt_to_txt('CrowdHuman/annotation_val_coco_style.json',
                   'running_log/YOLOX_640_last20noMosaic/last_epoch_evalresult.json',
                   'coco_val_dt')