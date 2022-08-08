import torch
import torch.nn as nn
from torch import tensor as t
from utility.loss import BCElossAmp

loss = BCElossAmp()

a = t([1,0,1,0], dtype=torch.float)
b = t([1,1,0,0], dtype=torch.float)

print(loss(a,b))