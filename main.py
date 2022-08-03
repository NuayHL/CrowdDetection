import torch
import torch.nn as nn

from utility.loss import SmoothL1, loss_dict_to_str

a = {'loss':0.3432, 'loss1':3425.3, 'loss2':352346}

print(loss_dict_to_str(a))