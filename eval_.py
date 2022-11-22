import os
import sys
import numpy as np
from odcore.engine.eval import Eval
from config import get_default_cfg
from odcore.args import get_eval_args_parser
from modelzoo.build_models import BuildModel

def main(args):
    cfg = get_default_cfg()
    cfg.merge_from_files(args.conf_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    builder = BuildModel(cfg)
    model = builder.build()
    model.set(args, 0)
    assert args.type == 'mr'
    args.force_eval = True
    iou_thres = np.arange(0.6, 0.8, 0.01).astype(float).tolist()
    summary = []
    summary_dict = dict()
    for step, iou_thre in enumerate(iou_thres):
        print('./n STEP[%d/%d]' % (step+1, len(iou_thres)))
        model.nms.iou_thres = iou_thre
        eval = Eval(cfg, args, model, 0)
        iou_id = str('%.2f' % iou_thre)
        val_result = eval.eval(record_name='IoU Threshold='+iou_id)
        summary.append(val_result)
        summary_dict[iou_id] = val_result

    for i in summary_dict.items():
        print(*i)

    summary = np.array(summary)
    best_iou_i, _ = summary.argmax(axis=0)
    print('Best Iou Threshold: %.2f' % iou_thres[best_iou_i])
    print('Best AP: %.6f' % summary[best_iou_i,0])

if __name__ == "__main__":
    args = get_eval_args_parser().parse_args()
    main(args)
