python -u train.py --conf-file cfgs/yolox_coco\
                   --device 0 --batch-size 32 --accumu 1 --workers 8 --eval-interval 10
echo " "
echo "Press AnyKey to Continue"
read -n 1
