import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
from odcore.engine.infer import Infer
from config import get_default_cfg
from odcore.args import get_args_parser
from modelzoo.build_models import BuildModel


def main(args):
    cfg = get_default_cfg()
    cfg.merge_from_file(args.conf_file)

    builder = BuildModel(cfg)
    model = builder.build()
    model.set(args, 0)
    infer = Infer(cfg, args, model, 0)
    result = infer(args.img)
    print(result)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
    a = 'IMG/IMG_20220608_140931.jpg'