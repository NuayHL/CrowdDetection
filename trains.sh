python -u train.py --conf-file YOLOv3_step.yaml\
                   --fine-tune running_log/YOLOv3_step1_finetune/last_epoch.pth\
                   --device 0 --batch-size 8 --accumu 4 --workers 8 --eval-interval 20
echo " "
echo "Press AnyKey to Continue"
read -n 1
