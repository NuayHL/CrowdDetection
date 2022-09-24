import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
from odcore.engine.infer import Infer
from config import get_default_cfg
from odcore.args import get_infer_args_parser
from odcore.utils.visualization import show_bbox
from modelzoo.build_models import BuildModel


def main(args):
    assert os.path.exists(args.img),'Invalid image path'
    cfg = get_default_cfg()
    cfg.merge_from_files(args.conf_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    builder = BuildModel(cfg)
    model = builder.build()
    model.set(args, 0)
    infer = Infer(cfg, args, model, 0)
    result, img = infer(args.img)
    result = result[0].to_ori_label()
    for i in result[:,4:]:
        print('score:', i[0],' category:',int(i[1]))
    if args.labeled:
        score = result[:, 4:]
    else:
        score = None
    show_bbox(img, result[:,:4], type='x1y1x2y2', score=score )


if __name__ == "__main__":
    args = get_infer_args_parser().parse_args()
    main(args)