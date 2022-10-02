import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
from odcore.engine.infer import Infer as _Infer
from config import get_default_cfg
from odcore.args import get_infer_args_parser
from odcore.utils.visualization import show_bbox, _add_bbox_img, printImg
from modelzoo.build_models import BuildModel


class Infer():
    def __init__(self, cfg, args, device):
        self.cfg = cfg
        self.device = device
        builder = BuildModel(cfg)
        model = builder.build()
        model.set(args, self.device)
        self.infer = _Infer(cfg, args, model, self.device)

    def __call__(self, img, img_only=False):
        result, img = self.infer(args.img)
        result = result[0].to_ori_label()
        if not img_only:
            return result, img
        else:
            img = _add_bbox_img(img, result, type='x1y1x2y2')
            return img

def main(args):
    assert os.path.exists(args.img),'Invalid image path'
    cfg = get_default_cfg()
    cfg.merge_from_files(args.conf_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    infer = Infer(cfg, args, 0)
    # ------------------------------------------------------------------------------------------------------------------
    result, img = infer(args.img, img_only=False)
    for i in result[:,4:]:
        print('score:', i[0],' category:',int(i[1]))
    if args.labeled:
        score = result[:, 4:]
    else:
        score = None
    show_bbox(img, result[:,:4], type='x1y1x2y2', score=score )
    # ------------------------------------------------------------------------------------------------------------------
    # img = infer(args.img, img_only=True)
    # printImg(img)

if __name__ == "__main__":
    args = get_infer_args_parser().parse_args()
    main(args)