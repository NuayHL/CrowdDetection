python -u train.py --conf-file cfgs/yolox_anchor3\
                   --device 0 --batch-size 1 --accumu 1 --workers 1 --eval-interval 10 --save-interval 50

echo " "
echo "Press AnyKey to Continue"
read -n 1
