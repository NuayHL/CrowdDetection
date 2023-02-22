## A One-stage Crowd ObjectDetection

![val_1_dt](imgs/val_1_dt.png)

This project aims to reproduce the YOLO series via pytorch

### Environment Setting
- Base Enviroment
```
pip install -r requirments.txt
```
- Pytorch Install

    check https://pytorch.org/get-started/locally/ for pytorch install


- odcore Install (Providing training, evaluating, inferencing, etc backend)
  
  https://github.com/NuayHL/odcore

  Note: This is my private Repo
```
git clone https://github.com/NuayHL/odcore.git
```

### Simple Inferencing 
- Downloading Model Weight: YOLOv3_640.pth

    https://drive.google.com/file/d/1-y_pMTb3lMLTr4Clw2TTQ5pYi_g--WeP/view?usp=sharing


- Infer through following command
```
python infer.py --ckpt-file YOLOv3_640.pth --conf-file cfgs/yolov3 --img imgs/val_1.png
```
- Replace the image name to infer other images

### Model Training

- Download **CrowdHuman** dataset
    
    https://www.crowdhuman.org/


- Rearrange the dataset as following structure
```
CrowdDetection
      |--CrowdHuman
            |--Images_train
            |      |-- All training images
            |--Images_val    
            |      |-- All val images
            |--annotation_train.odgt
            |--annotation_val.odgt
```
- Run `utility/tran_CrowdHuman.py` at root path
- Run command
  
```
python train.py --conf-file cfgs/yolox --batch-size 32 --workers 8 --device 0 --eval-interval 10 --save-interval 50
```
### Model Evaluation
**CrowdHuman dataset required*
```
python eval.py --conf-file cfgs/yolo_v4 --ckpt-file best_epoch.pth --workers 8 --batch-size 32 --type mip --force-eval
```
Replace the --ckpt-file and its correspond config yaml file for custom evaluation

If you have complete exp dir under the running_dir (exmple: running_log/rYOLOv4), you can start eval the .pth file
(example: best_epoch.pth) with *eval.sh* 

```
./eval.sh rYOLOv4 best_epoch mip
```

#### Evaluation type

- This repo has integrated with 3 evaluation metrics: 
  - `--type mip`: CrowdHuman AP + MR + JI (used as final evaluation)
    - https://github.com/megvii-model/CrowdDetection/tree/master/evaluate
  - `--type coco`: COCO (optional used during training)
    - https://github.com/cocodataset/cocoapi
  - `--type mr`: COCO based MR (optional used during training)
    - https://github.com/TencentYoutuResearch/PedestrianDetection-NohNMS/blob/main/detectron2/evaluation/crowdhuman_evaluation.py



### Loss visualization
```
python drawloss.py --loss-log exp_loss.log
```
- Replace the --loss-log to other _loss.log files to visualize them


*Copyright© NuayHL 2022. All right reserved*
