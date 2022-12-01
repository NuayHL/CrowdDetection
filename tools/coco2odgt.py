import os

import json
import cv2
from collections import defaultdict

ori_gtfile = 'CrowdHuman/annotation_val.odgt'

def get_odgt_dict():
    images = dict()

    print("begin convert val dataset annotations to dict format")
    val_imgs = load_json_lines(ori_gtfile)
    id = 0
    for img in val_imgs:
        id += 1
        h, w, _ = (cv2.imread("CrowdHuman/Images_val/"+img["ID"]+".jpg")).shape
        images[img["ID"]] = {"width":w, "height":h, 'gtboxes': list()}
        for bbox in img["gtboxes"]:
            if bbox["tag"] == "mask":
                continue
            else:
                bbox["tag"] = 1
            if 'extra' in bbox:
                if 'ignore' in bbox['extra'] and bbox['extra']['ignore'] != 0:
                    continue
            bbox['box'] = bbox['fbox']
            del bbox['fbox']
            del bbox['extra']
            if 'vbox' in bbox: del bbox['vbox']
            if 'hbox' in bbox: del bbox['hbox']
            if 'head_attr' in bbox: del bbox['head_attr']
            images[img["ID"]]['gtboxes'].append(bbox)
        progressbar(float(id/4370))

    json.dump(images, open("CrowdHuman/annotation_val_dict_style.json",'w'))
    return images

def coco2odgt_det(coco_det_path):
    """
    odgt_det format:
    [
        {
        "ID": image_id
        "dtboxes": [{"box":,
                    "score":}]
        }
    ]
    """
    if not os.path.exists("CrowdHuman/annotation_val_dict_style.json"):
        gt_dict = get_odgt_dict()
    else:
        with open("CrowdHuman/annotation_val_dict_style.json") as f:
            gt_dict = json.load(f)

    file_name = os.path.splitext(coco_det_path)[0] + '.odgt'
    odgt_det = defaultdict(list)
    with open(coco_det_path) as f:
        coco_det = json.load(f)
    for det_box in coco_det:
        img_id = det_box['image_id']
        box = det_box['bbox']
        score = det_box['score']
        odgt_det[img_id].append({'box':box,'score':score})
    odgt_det = list(odgt_det.items())
    odgt_det = [{'ID':det[0],'dtboxes':det[1], 'width':gt_dict[det[0]]['width'], 'height':gt_dict[det[0]]['height'],
                 'gtboxes':gt_dict[det[0]]['gtboxes']} for det in odgt_det]
    save_json_lines(odgt_det, file_name)
    return file_name

def load_json_lines(fpath):
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records

def save_json_lines(content,fpath):
    with open(fpath,'w') as fid:
        for db in content:
            line = json.dumps(db)+'\n'
            fid.write(line)

def progressbar(percentage, endstr='', barlenth=20):
    if int(percentage)==1: endstr +='\n'
    print('\r[' + '>' * int(percentage * barlenth) +
          '-' * (barlenth - int(percentage * barlenth)) + ']',
          format(percentage * 100, '.1f'), '%', end=' '+endstr)

if __name__ == "__main__":
    path = os.getcwd()
    os.chdir(os.path.join(path, '..'))
    coco2odgt_det('running_log/YOLOX_ori_1/best_epoch_evalresult.json')
