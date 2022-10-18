from torch import nn as nn

from modelzoo.common import BasicBlock, Bottleneck
from modelzoo.backbone.build import BackboneRegister

# Part from:
#     https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/utils.py

class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        p3 = self.layer2(x)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)

        return p3, p4, p5


@BackboneRegister.register
def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.p3c = 128
    return model

@BackboneRegister.register
def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    model.p3c = 128
    return model

@BackboneRegister.register
def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model.p3c = 512
    return model

@BackboneRegister.register
def resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    model.p3c = 512
    return model

@BackboneRegister.register
def resnet152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    model.p3c = 512
    return model
