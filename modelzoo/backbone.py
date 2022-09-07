import torch
import torch.nn as nn

from modelzoo.common import conv_nobias_bn_lrelu
from modelzoo.common import DarkResidualBlock
from modelzoo.common import make_layers

def build_backbone(name):
    '''return backboneClass, output p3c'''
    if name == 'darknet53':
        return Darknet53, 256
    elif name == 'resnet18':
        return resnet18, 128
    elif name == 'resnet34':
        return resnet34, 128
    elif name == 'resnet50':
        return resnet50, 512
    elif name == 'resnet18':
        return resnet101, 512
    elif name == 'resnet18':
        return resnet152, 512
    elif name == 'cspdarknet':
        return CSPDarknet, None
    else:
        raise NotImplementedError('No backbone named %s' % (name))

class Darknet53(nn.Module):
    def __init__(self, res_block=DarkResidualBlock):
        super(Darknet53, self).__init__()

        self.conv1 = conv_nobias_bn_lrelu(3, 32)
        self.conv2 = conv_nobias_bn_lrelu(32, 64, stride=2)
        self.residual_block1 = make_layers(1, res_block, 64)
        self.conv3 = conv_nobias_bn_lrelu(64, 128, stride=2)
        self.residual_block2 = make_layers(2, res_block, 128)
        self.conv4 = conv_nobias_bn_lrelu(128, 256, stride=2)
        self.residual_block3 = make_layers(8, res_block, 256)
        self.conv5 = conv_nobias_bn_lrelu(256, 512, stride=2)
        self.residual_block4 = make_layers(8, res_block, 512)
        self.conv6 = conv_nobias_bn_lrelu(512, 1024, stride=2)
        self.residual_block5 = make_layers(4, res_block, 1024)

    def forward(self,x):
        ''' p3 channel: 256 '''
        x = self.conv1(x)
        x = self.residual_block1(self.conv2(x))
        x = self.residual_block2(self.conv3(x))
        p3 = self.residual_block3(self.conv4(x))
        p4 = self.residual_block4(self.conv5(p3))
        p5 = self.residual_block5(self.conv6(p4))
        return p3, p4, p5

from modelzoo.common import BasicBlock, Bottleneck
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

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

from modelzoo.common import BaseConv, DWConv, CSPLayer, Focus, SPPBottleneck
class CSPDarknet(nn.Module):
    def __init__(self, depth=1.0, width=1.0, depthwise=False, act='silu'):
        super(CSPDarknet, self).__init__()
        base_channels = int(width * 64)
        base_depth = max(round(depth * 3), 1)
        Conv = DWConv if depthwise else BaseConv
        self.p3c = base_channels * 4

        self.stem = Focus(3, base_channels, kernel_size=3, act=act)

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2, base_channels * 2,
                n=base_depth, depthwise=depthwise, act=act
            ),
        )
        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4, base_channels * 4,
                n=base_depth * 3, depthwise=depthwise, act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8, base_channels * 8,
                n=base_depth * 3, depthwise=depthwise, act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16, base_channels * 16, n=base_depth,
                shortcut=False, depthwise=depthwise, act=act,
            ),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return p3, p4, p5

if __name__ == "__main__":
    dummy_input = torch.ones([1,3,64,64])
    model = CSPDarknet()
    output = model(dummy_input)
    print(output[0].shape,output[1].shape,output[2].shape)


