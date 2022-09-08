python -u train.py --conf-file cfgs/yolo\
                   --fine-tune YOLOv3_640.pth\
                   --device 0 --batch-size 4 --accumu 1 --workers 4 --eval-interval 20
echo " "
echo "Press AnyKey to Continue"
read -n 1
