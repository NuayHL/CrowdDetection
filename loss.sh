exp_name=$1
echo "Exp name: "$exp_name""
python drawloss.py --loss-log "running_log/"$exp_name"/"$exp_name"_loss.log"