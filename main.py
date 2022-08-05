import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
from odcore.engine.infer import Infer
from config import get_default_cfg
from odcore.args import get_args_parser
from modelzoo.build_models import BuildModel

cfg = get_default_cfg()
cfg.merge_from_file('test_config.yaml')
args = get_args_parser().parse_args()
builder = BuildModel(cfg)
model = builder.build()
model.set(args, 1)
infer = Infer(cfg, args, model, 1)
result = infer('IMG/IMG_20220608_140931.jpg')
print(result)