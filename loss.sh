path=$1
exp_name=${path##*/}
echo "Exp name: "$exp_name""
python drawloss.py --loss-log "running_log/"$path"/"$exp_name"_loss.log"
echo " "
echo "Press AnyKey to Continue"
read -n 1