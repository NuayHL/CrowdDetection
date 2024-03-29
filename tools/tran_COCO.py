import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))

import json

def progressbar(percentage, endstr='', barlenth=20):
    if int(percentage)==1: endstr +='\n'
    print('\r[' + '>' * int(percentage * barlenth) +
          '-' * (barlenth - int(percentage * barlenth)) + ']',
          format(percentage * 100, '.1f'), '%', end=' '+endstr)

def modify_categories(filepath, outputname):
    with open("../odcore/data/categories_coco.json", 'r') as fp:
        coco_cat = json.load(fp)
    trans_dict = {}
    for i, classname in enumerate(coco_cat):
        trans_dict[int(classname['id'])] = i
        classname['id'] = i

    with open(filepath) as f:
        annos = json.load(f)
    anno_len = float(len(annos['annotations']))
    for i, anno in enumerate(annos['annotations']):
        anno['category_id'] = trans_dict[anno['category_id']]
        progressbar((i+1)/anno_len, barlenth=40)
    annos['categories'] = coco_cat
    with open(os.path.join(os.path.dirname(filepath), outputname),'w') as f:
        json.dump(annos, f)

if __name__ == '__main__':
    modify_categories('../COCO/instances_val2017.json','val2017_coco.json')
    modify_categories('../COCO/instances_train2017.json','train2017_coco.json')
