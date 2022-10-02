path=$1
img_name=$2
exp_name=${path##*/}
echo "Exp name: "$exp_name""
echo "Image name: "$img_name""
python -u infer.py --conf-file "running_log/"$path"/"$exp_name"_cfg.yaml" \
                   --ckpt-file "running_log/"$path"/last_epoch.pth" \
                   --source $img_name
echo " "
echo "Press AnyKey to Continue"
read -n 1