path=$1
pth_name=$2
exp_name=${path##*/}
evaltype=$3

python eval.py --conf-file "running_log/"$exp_name"/"$exp_name"_cfg.yaml"\
               --ckpt-file "running_log/"$exp_name"/"$pth_name".pth"\
               --batch-size 32 --workers 16 --forward-func $evaltype