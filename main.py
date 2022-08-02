import torch
import torch.nn as nn

from utility.loss import SmoothL1

a = torch.tensor((1,0,1,0)).float()
b = torch.tensor((1,1,0,0)).float()

loss = SmoothL1()
c = loss(a,b)
print(c)