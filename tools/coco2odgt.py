import os
path = os.getcwd()
os.chdir(os.path.join(path, '..'))

import json
from collections import defaultdict
from evaluate.misc_utils import save_json_lines

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
    file_name = os.path.splitext(coco_det_path)[0] + '.odgt'

    odgt_det = defaultdict(list)
    with open(coco_det_path) as f:
        coco_det = json.load(f)
    for det_box in coco_det:
        img_id = det_box['image_id']
        box = det_box['bbox']
        # box[2] += box[0]
        # box[3] += box[1]
        score = det_box['score']
        odgt_det[img_id].append({'box':box,'score':score})
    odgt_det = list(odgt_det.items())
    odgt_det = [{'ID':det[0],'dtboxes':det[1]} for det in odgt_det]
    save_json_lines(odgt_det, file_name)
    return file_name

if __name__ == "__main__":
    coco2odgt_det('running_log/YOLOX_cbam_caffn1_anchor3/epoch_294_evalresult.json')
