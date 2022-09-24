import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
import json
import cv2

def progressbar(percentage, endstr='', barlenth=20):
    if int(percentage)==1: endstr +='\n'
    print('\r[' + '>' * int(percentage * barlenth) +
          '-' * (barlenth - int(percentage * barlenth)) + ']',
          format(percentage * 100, '.1f'), '%', end=' '+endstr)

def odgt2coco(filepath, outputname, type, limit_img=30000):
    """
    :param filepath: origin odgt file
    :param outputname: the name of output json file
    :param type: "train" or "val"
    :return: None

    About the CrowdHuman Dataset:
    CrowdHuman contains 15000 training imgs,
                        4370 validation imgs,
                        5000 testing imgs.

    About COCO format:{
        "info":{"year"}
        "images":[image]
        "annotations":[annotation]
        "categories":{}
    }
    image{
        "id": int,
        "width": int,
        "height": int,
        "file_name": str
    }
    annotation{
        "id": int,
        "image_id": int,
        "category_id": int,
        "bbox": [x1,y1,width,height] (fbox)
        "segmentation": points of vbox
        "area": area of bbox
        "iscrowd": 0 or 1
    }
    categories[{
        "id": int,
        "name": str,
        "supercategory": str,
    }]
    """
    assert type in ["train", "val"], "The type should be \'train\' or \'val\'"
    if type == "train": num_imgs = 15000
    else: num_imgs = 4370
    info = dict()
    images = []
    annotations = []
    categories = [{"supercategory": "person", "id": 0, "name": "person"}]

    info["year"] = 2018
    print("begin convert %s dataset annotations to coco format"%type)
    with open(filepath) as f:
        id = 0
        bbox_id=0
        for sample in f:
            id += 1
            if id >= limit_img: break
            img = json.loads(sample)
            h, w, _ = (cv2.imread("CrowdHuman/Images_"+type+"/"+img["ID"]+".jpg")).shape
            img_info = {"id":img["ID"], "width":w, "height":h, "file_name":img["ID"]+'.jpg'}
            images.append(img_info)
            for bbox in img["gtboxes"]:
                # find the ignore area
                if bbox["tag"] == "mask": categories_id = -1
                else: categories_id = 0
                if "ignore" in bbox["extra"].keys() and bbox["extra"]["ignore"] == 1: categories_id = -1

                # if it is val_dataset, then do not add to eval
                if categories_id == -1 and type == "val": continue

                bbox_id += 1
                # fbox area
                area = bbox['fbox'][2] * bbox['fbox'][3]
                # segmentation
                segmentation = []
                segmentation.append(bbox['vbox'][0])
                segmentation.append(bbox['vbox'][1])
                segmentation.append(bbox['vbox'][0] + bbox['vbox'][2])
                segmentation.append(bbox['vbox'][1])
                segmentation.append(bbox['vbox'][0] + bbox['vbox'][2])
                segmentation.append(bbox['vbox'][1] + bbox['vbox'][3])
                segmentation.append(bbox['vbox'][0])
                segmentation.append(bbox['vbox'][1] + bbox['vbox'][3])
                segmentation = [segmentation]

                bbox_info={"id":bbox_id,"image_id":img["ID"],"category_id": categories_id, "segmentation": segmentation,
                           "bbox":bbox["fbox"], "area":area, "iscrowd":0}
                annotations.append(bbox_info)
            progressbar(float(id/num_imgs))

    output = {"info":info, "images":images, "annotations":annotations, "categories":categories}
    json.dump(output, open("CrowdHuman/"+outputname+".json",'w'))

if __name__ == '__main__':
    odgt2coco("CrowdHuman/annotation_val.odgt", "annotation_val_coco_style", "val")
    odgt2coco("CrowdHuman/annotation_train.odgt","annotation_train_coco_style","train")

