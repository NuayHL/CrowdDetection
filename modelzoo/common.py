import torch
import torch.nn as nn

def conv_batch(ic, oc, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(ic, oc, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oc),
        nn.LeakyReLU(inplace=True))

def make_layers(num, block, *args, **kwargs):
    layers = []
    for i in range(num):
        layers.append(block(*args, **kwargs))
    return nn.Sequential(*layers)

class DarkResidualBlock(nn.Module):
    def __init__(self, ic):
        super(DarkResidualBlock, self).__init__()
        reduced_channels = int(ic/2)
        self.conv1 = conv_batch(ic, reduced_channels, kernel_size=1, padding=0)
        self.conv2 = conv_batch(reduced_channels, ic, kernel_size=3, padding=1)

    def forward(self,x):
        residual = x
        x = self.conv2(self.conv1(x))
        return x+residual