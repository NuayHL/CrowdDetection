python -u train.py --conf-file cfgs/yolox_free_s2_ignored\
                   --device 0 --batch-size 32 --accumu 1 --workers 16 --eval-interval 10 --save-interval 50

echo " "
echo "Press AnyKey to Continue"
read -n 1
