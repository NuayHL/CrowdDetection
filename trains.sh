python -u train.py --conf-file cfgs/yolox\
                   --device 0 --batch-size 32 --accumu 1 --workers 8 --eval-interval 20
echo " "
echo "Press AnyKey to Continue"
read -n 1
