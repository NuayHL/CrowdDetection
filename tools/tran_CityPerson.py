import os
path = os.getcwd()
os.chdir(os.path.join(path, '..'))
import scipy.io
import json
import cv2

def Cityperson2coco(filepaths, anno_type='train'):
    '''
    :param filepaths: pathfile list
    :param outputname: the name of output json file
    :return: None
    About the WiderPerson Dataset:
    CityPerson contains 2975 training imgs with annotation,
                        500 validation imgs with annotation,

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
        "bbox": [x,y,width,height] (fbox),
        "iscrowd": 0 or 1
        "area": full bbox area
        "height": height of bbox
        "vis_ratio": vis_area / full_area
    }
    categories[{
        "id": int,
        "name": str,
        "supercategory": str,
    }]
    '''
    assert anno_type in ['val', 'train']

    info = dict()
    images = []
    annotations = []
    categories = [{"supercategory": "person", "id": 0, "name": "person"}]

    info["year"] = 2018
    print("begin convert %s to coco format" % filepaths)

    file_name = os.path.basename(os.path.splitext(filepaths)[0])
    output_name = file_name + '_coco_style.json'
    dict_name = file_name + '_aligned'
    imageinfos = scipy.io.loadmat(filepaths)[dict_name][0]

    images_path = "CityPerson"
    if 'train' in dict_name:
        images_path = os.path.join(images_path, 'train')
    elif 'val' in dict_name:
        images_path = os.path.join(images_path, 'val')

    num_imgs = len(imageinfos)
    for idx, imageinfo in enumerate(imageinfos):
        imageinfo = imageinfo[0][0]
        city_name = str(imageinfo[0])[2:][:-2]
        image_name = str(imageinfo[1])[2:][:-2]
        image_annos = imageinfo[2].astype(int).tolist()
        image_id = os.path.splitext(image_name)[0]
        real_image_path = os.path.join(city_name, image_name)
        h, w, _ = (cv2.imread(os.path.join(images_path, real_image_path))).shape
        img_info = {"id":image_id, "width":w, "height":h, "file_name":real_image_path}
        images.append(img_info)

        for anno in image_annos:
            bbox_cat = anno[0]
            categories_id = 0 if bbox_cat in [1,2,3,4] else -1
            if categories_id == 0 and anno_type == 'val':
                continue
            iscrowd = 1 if bbox_cat == 5 else 0
            bbox_id = anno[5]
            bbox_full = anno[1:5]
            bbox_vis = anno[6:]
            area = float(bbox_full[2] * bbox_full[3])  # bbox_area
            vis_area = float(bbox_vis[2] * bbox_vis[3])  # vis_bbox_area

            bbox_info = {"id":bbox_id,"image_id":image_id, "category_id":categories_id,
                         "bbox":bbox_full, "area":area, "iscrowd": iscrowd, "height": bbox_full[3],
                         "vbox":bbox_vis, 'vis_ratio': vis_area/area}
            annotations.append(bbox_info)
        progressbar(float((idx+1)/num_imgs))

    output = {"info":info, "images":images, "annotations":annotations, "categories":categories}
    json.dump(output, open(os.path.join("CityPerson/", output_name),'w'))

def progressbar(percentage, endstr='', barlenth=20):
    if int(percentage)==1: endstr +='\n'
    print('\r[' + '>' * int(percentage * barlenth) +
          '-' * (barlenth - int(percentage * barlenth)) + ']',
          format(percentage * 100, '.1f'), '%', end=' '+endstr)

if __name__ == '__main__':
    Cityperson2coco("CityPerson/anno_train.mat", "train")
    Cityperson2coco("CityPerson/anno_val.mat", "val")
