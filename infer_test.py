import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
import matplotlib.pyplot as plt
import numpy as np
from odcore.engine.infer import Infer
from odcore.utils.visualization import stack_img, generate_hot_bar
from config import get_default_cfg
from odcore.args import get_infer_args_parser
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
    result, img = infer(args.img, "_test_hot_map")
    sum_result = []
    fig, ax = plt.subplots(3,4)
    for id, level in enumerate(result):
        for il, fm in enumerate(level):
            fm = fm.numpy()
            sum_result.append(fm)
            ax[id][il].imshow(fm)
            ax[id][il].axis('off')
    plt.show()
    sum_result = stack_img(sum_result,(3,4))
    bar = generate_hot_bar(1.0, 0.0, sum_result.shape[0])
    sum_result = np.concatenate([sum_result, bar], axis=1)
    fig, ax = plt.subplots()
    ax.imshow(sum_result)
    ax.axis('off')
    plt.show()



if __name__ == "__main__":
    args = get_infer_args_parser().parse_args()
    main(args)