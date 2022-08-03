import torch
import torch.nn as nn

from utility.loss import SmoothL1

softmax = nn.Softmax(dim=1)
a = torch.ones((3,4,3))
print(a)
a = softmax(a)
print(a)