import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
os.chdir(os.path.join(path, '..'))
import torch
from modelzoo.build_models import numofParameters
from modelzoo.neck.build import build_neck
p3 = torch.rand((1, 256, 32, 32))
p4 = torch.rand((1, 512, 16, 16))
p5 = torch.rand((1, 1024, 8, 8))

neck = build_neck('retina_neck')
fpn = neck(256)
print("Num of Parameters: %.2fM" % (numofParameters(fpn) / 1e6))
result = fpn(p3, p4, p5)
for i in result:
    print(i.shape)