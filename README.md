## Crowd ObjectDetection based on YOLO

#### Environment Setting
- Base Enviroment
```
pip install -r requirments.txt
```
- Pytorch Install

    check https://pytorch.org/get-started/locally/ for pytorch install
#### Simple Inferencing 
- Downloading Model Weight: YOLOv3_640.pth

    https://drive.google.com/file/d/1-y_pMTb3lMLTr4Clw2TTQ5pYi_g--WeP/view?usp=sharing


- Infer through following command
```
python infer.py --ckpt-file YOLOv3_640.pth --conf-file YOLOv3.yaml --img img_fountain.jpg
```
- Replace the image name to infer other images

#### Model Training

- Download **CrowdHuman** dataset
    
    https://www.crowdhuman.org/


- Rearrange the dataset as following structure
```
CrowdDetection
      |--CrowdHuman
            |--Images_train
                  |-- All training images
            |--Images_val    
                  |-- All val images
            |--annotation_train.odgt
            |--annotation_val.odgt
```
- Run `tran_CrowdHuman.py`
- Run command
  
    `python train.py --conf-file YOLOv3.yaml --batch-size 32 --workers 8 --device 0`
#### Model Evaluation

- CrowdHuman dataset required ! 
```
python eval.py --conf-file YOLOv3.yaml --ckpt-file YOLOv3_640.pth --workers 8 --batch-size 32 --device 0 
```
- Replace the --ckpt-file to other .pth files to evaluate them

#### Loss visualization
```
python drawloss.py --loss-log YOLOv3_siou_loss.log
```
- Replace the --loss-log to other _loss.log files to visualize them


*Copyright© Haoyuan Liu 2022. All right reserved*
