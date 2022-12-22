import os
path = os.getcwd()
os.chdir(os.path.join(path, '..'))
import json
import cv2

Ignore_Type = ['person-group-far-away',
               'rider+vehicle-group-far-away']

Pass_Type = ['bicycle-group', 'buggy-group', 'motorbike-group',
             'scooter-group', 'tricycle-group', 'wheelchair-group',
             'motorbike', 'buggy', 'scooter', 'tricycle', 'bicycle', 'wheelchair']

def ECP2coco(label_path, anno_type='train'):
    '''
    :param filepaths: pathfile list
    :param outputname: the name of output json file
    :return: None
    About the WiderPerson Dataset:
    ECP contains :

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

    basedri = os.path.dirname(label_path)
    output_name = os.path.join(basedri, anno_type + '_coco_style.json')

    info = dict()
    images = []
    annotations = []
    categories = [{"supercategory": "person", "id": 0, "name": "pedestrian"},
                  {"supercategory": "person", "id": 1, "name": "rider"}]

    info["year"] = 2018
    print("begin convert %s to coco format" % label_path)

    categories_set = []

    bbox_id_count = 0
    city_names = os.listdir(label_path)
    for city in city_names:
        city_path = os.path.join(label_path, city)
        city_images_jsons = os.listdir(city_path)
        num_imgs = len(city_images_jsons)
        for idx, json_path in enumerate(city_images_jsons):
            with open(os.path.join(city_path, json_path)) as f:
                imageinfo = json.load(f)
            h, w = imageinfo['imageheight'], imageinfo['imagewidth']
            image_id = os.path.splitext(json_path)[0]
            real_image_path = os.path.join(city, image_id+'.png')
            img_info = {"id":image_id, "width":w, "height":h, "file_name":real_image_path}
            images.append(img_info)

            image_annos = imageinfo['children']
            for anno in image_annos:
                bbox_id = bbox_id_count
                bbox_id_count += 1
                categories_name = anno['identity']
                if categories_name in Pass_Type:
                    continue
                elif categories_name == 'pedestrian':
                    categories_id = 0
                elif categories_name == 'rider':
                    categories_id = 1
                elif categories_name in Ignore_Type:
                    categories_id = -1
                else:
                    raise NameError

                categories_set.append(categories_name)
                x1 = float(anno['x0'])
                y1 = float(anno['y0'])
                w = float(anno['x1'] - anno['x0'])
                h = float(anno['y1'] - anno['y0'])
                bbox_full = [x1,y1,w,h]
                area = w * h
                vis_ratio = 1.0
                if 'tag' in anno:
                    if 'occluded>10' in anno['tag'] or 'truncated>10' in anno['tag']:
                        vis_ratio = 0.9
                    elif 'occluded>40' in anno['tag'] or 'truncated>40' in anno['tag']:
                        vis_ratio = 0.6
                    elif 'occluded>80' in anno['tag'] or 'truncated>80' in anno['tag']:
                        vis_ratio = 0.2

                bbox_info = {"id":bbox_id,"image_id":image_id, "category_id":categories_id,
                             "bbox":bbox_full, "area":area, "iscrowd": 0, "height": bbox_full[3],
                             'vis_ratio': vis_ratio}
                annotations.append(bbox_info)
            progressbar(float((idx+1)/num_imgs), endstr=city)

    print(set(categories_set))
    output = {"info":info, "images":images, "annotations":annotations, "categories":categories}
    json.dump(output, open(output_name,'w'))

def progressbar(percentage, endstr='', barlenth=20):
    if int(percentage)==1: endstr +='\n'
    print('\r[' + '>' * int(percentage * barlenth) +
          '-' * (barlenth - int(percentage * barlenth)) + ']',
          format(percentage * 100, '.1f'), '%', end=' '+endstr)

if __name__ == '__main__':
    ECP2coco("ECP/day/labels/train", "train")
    ECP2coco("ECP/day/labels/val", "val")
