import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "mish":
        module = nn.Mish(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

def make_layers(num, block, *args, **kwargs):
    layers = []
    for i in range(num):
        layers.append(block(*args, **kwargs))
    return nn.Sequential(*layers)

def conv_nobias_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, act = 'silu'):
        super(DWConv, self).__init__()
        self.layer_conv = BaseConv(in_channels, in_channels, kernel_size, stride, groups=in_channels, act=act)
        self.depth_conv = BaseConv(in_channels, out_channels, kernel_size=1, stride=1, groups=1, act=act)
    def forward(self, x):
        return self.depth_conv(self.layer_conv(x))


def conv_nobias_bn_silu(ic, oc, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(ic, oc, kernel_size, stride, padding, bias=False, groups=groups),
        nn.BatchNorm2d(oc),
        nn.SiLU(inplace=True))

## Darknet
def conv_nobias_bn_lrelu(ic, oc, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(ic, oc, kernel_size, stride, padding, bias=False, groups=groups),
        nn.BatchNorm2d(oc),
        nn.LeakyReLU(inplace=True))

class DarkResidualBlock(nn.Module):
    def __init__(self, ic):
        super(DarkResidualBlock, self).__init__()
        reduced_channels = int(ic/2)
        self.conv1 = conv_nobias_bn_lrelu(ic, reduced_channels, kernel_size=1, padding=0)
        self.conv2 = conv_nobias_bn_lrelu(reduced_channels, ic, kernel_size=3, padding=1)

    def forward(self,x):
        residual = x
        x = self.conv2(self.conv1(x))
        return x+residual

## Resnet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

# RepVGGBlock is a basic rep-style block, including training and deploy status
# This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
from modelzoo.se_block import SEBlock
class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_nobias_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_nobias_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

# YOLOX
class Bottleneck_YOLO(nn.Module):
    def __init__(
        self, in_channels, out_channels, shortcut=True,
        expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, kernel_size, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,
        )
        return self.conv(x)

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""
    def __init__(
        self, in_channels, out_channels, n=1,
        shortcut=True, expansion=0.5, depthwise=False, act="silu"
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck_YOLO(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat([x_1, x_2], dim=1)
        return self.conv3(x)

class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

class YOLOv4_Common:
    # The following code in this class is modified from
    # https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/models.py
    class Mish(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x * (torch.tanh(torch.nn.functional.softplus(x)))
            return x

    class Upsample(nn.Module):
        def __init__(self):
            super(YOLOv4_Common.Upsample, self).__init__()

        def forward(self, x, target_size, inference=False):
            assert (x.data.dim() == 4)
            # _, _, tH, tW = target_size

            if inference:

                # B = x.data.size(0)
                # C = x.data.size(1)
                # H = x.data.size(2)
                # W = x.data.size(3)

                return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1). \
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3),
                           target_size[3] // x.size(3)). \
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
            else:
                return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')

    class Conv_Bn_Activation(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
            super().__init__()
            pad = (kernel_size - 1) // 2

            self.conv = nn.ModuleList()
            if bias:
                self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
            else:
                self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
            if bn:
                self.conv.append(nn.BatchNorm2d(out_channels))
            if activation == "mish":
                self.conv.append(YOLOv4_Common.Mish())
            elif activation == "relu":
                self.conv.append(nn.ReLU(inplace=True))
            elif activation == "leaky":
                self.conv.append(nn.LeakyReLU(0.1, inplace=True))
            elif activation == "linear":
                pass
            else:
                raise NotImplementedError

        def forward(self, x):
            for l in self.conv:
                x = l(x)
            return x

    class ResBlock(nn.Module):
        """
        Sequential residual blocks each of which consists of \
        two convolution layers.
        Args:
            ch (int): number of input and output channels.
            nblocks (int): number of residual blocks.
            shortcut (bool): if True, residual tensor addition is enabled.
        """

        def __init__(self, ch, nblocks=1, shortcut=True):
            super().__init__()
            self.shortcut = shortcut
            self.module_list = nn.ModuleList()
            for i in range(nblocks):
                resblock_one = nn.ModuleList()
                resblock_one.append(YOLOv4_Common.Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
                resblock_one.append(YOLOv4_Common.Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
                self.module_list.append(resblock_one)

        def forward(self, x):
            for module in self.module_list:
                h = x
                for res in module:
                    h = res(h)
                x = x + h if self.shortcut else h
            return x


class YOLOv7_common:
    def autopad(k, p=None):
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p

    class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
            super().__init__()
            act = nn.SiLU()
            self.conv = nn.Conv2d(c1, c2, k, s, YOLOv7_common.autopad(k, p), groups=g, bias=False)
            self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
            self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
                act if isinstance(act, nn.Module) else nn.Identity())

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

        def fuseforward(self, x):
            return self.act(self.conv(x))

    class Multi_Concat_Block(nn.Module):
        def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
            super().__init__()
            c_ = int(c2 * e)

            Conv = YOLOv7_common.Conv

            self.ids = ids
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.cv3 = nn.ModuleList(
                [Conv(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)]
            )
            self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

        def forward(self, x):
            x_1 = self.cv1(x)
            x_2 = self.cv2(x)

            x_all = [x_1, x_2]
            # [-1, -3, -5, -6] => [5, 3, 1, 0]
            for i in range(len(self.cv3)):
                x_2 = self.cv3[i](x_2)
                x_all.append(x_2)

            out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
            return out

    class Transition_Block(nn.Module):
        def __init__(self, c1, c2):
            super().__init__()
            Conv = YOLOv7_common.Conv
            self.cv1 = Conv(c1, c2, 1, 1)
            self.cv2 = Conv(c1, c2, 1, 1)
            self.cv3 = Conv(c2, c2, 3, 2)

            self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            # 160, 160, 256 => 80, 80, 256 => 80, 80, 128
            x_1 = self.mp(x)
            x_1 = self.cv1(x_1)

            # 160, 160, 256 => 160, 160, 128 => 80, 80, 128
            x_2 = self.cv2(x)
            x_2 = self.cv3(x_2)

            # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
            return torch.cat([x_2, x_1], 1)

    class RepConv(nn.Module):
        # Represented convolution
        # https://arxiv.org/abs/2101.03697
        def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=nn.SiLU(), deploy=False):
            super().__init__()
            self.deploy = deploy
            self.groups = g
            self.in_channels = c1
            self.out_channels = c2

            autopad = YOLOv7_common.autopad

            assert k == 3
            assert autopad(k, p) == 1

            padding_11 = autopad(k, p) - k // 2
            self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
                act if isinstance(act, nn.Module) else nn.Identity())

            if deploy:
                self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
            else:
                self.rbr_identity = (
                    nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
                self.rbr_dense = nn.Sequential(
                    nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                    nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
                )
                self.rbr_1x1 = nn.Sequential(
                    nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                    nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
                )

        def forward(self, inputs):
            if hasattr(self, "rbr_reparam"):
                return self.act(self.rbr_reparam(inputs))
            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(inputs)
            return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

        def get_equivalent_kernel_bias(self):
            kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
            kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
            return (
                kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
                bias3x3 + bias1x1 + biasid,
            )

        def _pad_1x1_to_3x3_tensor(self, kernel1x1):
            if kernel1x1 is None:
                return 0
            else:
                return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

        def _fuse_bn_tensor(self, branch):
            if branch is None:
                return 0, 0
            if isinstance(branch, nn.Sequential):
                kernel = branch[0].weight
                running_mean = branch[1].running_mean
                running_var = branch[1].running_var
                gamma = branch[1].weight
                beta = branch[1].bias
                eps = branch[1].eps
            else:
                assert isinstance(branch, nn.BatchNorm2d)
                if not hasattr(self, "id_tensor"):
                    input_dim = self.in_channels // self.groups
                    kernel_value = np.zeros(
                        (self.in_channels, input_dim, 3, 3), dtype=np.float32
                    )
                    for i in range(self.in_channels):
                        kernel_value[i, i % input_dim, 1, 1] = 1
                    self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
                kernel = self.id_tensor
                running_mean = branch.running_mean
                running_var = branch.running_var
                gamma = branch.weight
                beta = branch.bias
                eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

        def repvgg_convert(self):
            kernel, bias = self.get_equivalent_kernel_bias()
            return (
                kernel.detach().cpu().numpy(),
                bias.detach().cpu().numpy(),
            )

        def fuse_conv_bn(self, conv, bn):
            std = (bn.running_var + bn.eps).sqrt()
            bias = bn.bias - bn.running_mean * bn.weight / std

            t = (bn.weight / std).reshape(-1, 1, 1, 1)
            weights = conv.weight * t

            bn = nn.Identity()
            conv = nn.Conv2d(in_channels=conv.in_channels,
                             out_channels=conv.out_channels,
                             kernel_size=conv.kernel_size,
                             stride=conv.stride,
                             padding=conv.padding,
                             dilation=conv.dilation,
                             groups=conv.groups,
                             bias=True,
                             padding_mode=conv.padding_mode)

            conv.weight = torch.nn.Parameter(weights)
            conv.bias = torch.nn.Parameter(bias)
            return conv

        def fuse_repvgg_block(self):
            if self.deploy:
                return
            print(f"RepConv.fuse_repvgg_block")
            self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

            self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
            rbr_1x1_bias = self.rbr_1x1.bias
            weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

            # Fuse self.rbr_identity
            if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                            nn.modules.batchnorm.SyncBatchNorm)):
                identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups,
                    bias=False)
                identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
                identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
                identity_conv_1x1.weight.data.fill_(0.0)
                identity_conv_1x1.weight.data.fill_diagonal_(1.0)
                identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

                identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
                bias_identity_expanded = identity_conv_1x1.bias
                weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
            else:
                bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
                weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

            self.rbr_dense.weight = torch.nn.Parameter(
                self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
            self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

            self.rbr_reparam = self.rbr_dense
            self.deploy = True

            if self.rbr_identity is not None:
                del self.rbr_identity
                self.rbr_identity = None

            if self.rbr_1x1 is not None:
                del self.rbr_1x1
                self.rbr_1x1 = None

            if self.rbr_dense is not None:
                del self.rbr_dense
                self.rbr_dense = None

    def fuse_conv_and_bn(conv, bn):
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              groups=conv.groups,
                              bias=True).requires_grad_(False).to(conv.weight.device)

        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
        return fusedconv


class YOLOv8_common:
    def autopad(k, p=None, d=1):
        # kernel, padding, dilation
        # 对输入的特征层进行自动padding，按照Same原则
        if d > 1:
            # actual kernel-size
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        if p is None:
            # auto-pad
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p

    class SiLU(nn.Module):
        # SiLU激活函数
        @staticmethod
        def forward(x):
            return x * torch.sigmoid(x)

    class Conv(nn.Module):
        # 标准卷积+标准化+激活函数
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            default_act = YOLOv8_common.SiLU()
            autopad = YOLOv8_common.autopad
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            self.act = default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

        def forward_fuse(self, x):
            return self.act(self.conv(x))

    class Bottleneck(nn.Module):
        # 标准瓶颈结构，残差结构
        # c1为输入通道数，c2为输出通道数
        def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            Conv = YOLOv8_common.Conv
            self.cv1 = Conv(c1, c_, k[0], 1)
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    class C2f(nn.Module):
        # CSPNet结构结构，大残差结构
        # c1为输入通道数，c2为输出通道数
        def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
            super().__init__()
            Conv = YOLOv8_common.Conv
            self.c = int(c2 * e)
            self.cv1 = Conv(c1, 2 * self.c, 1, 1)
            self.cv2 = Conv((2 + n) * self.c, c2, 1)
            self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

        def forward(self, x):
            # 进行一个卷积，然后划分成两份，每个通道都为c
            y = list(self.cv1(x).split((self.c, self.c), 1))
            # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))

    class SPPF(nn.Module):
        # SPP结构，5、9、13最大池化核的最大池化。
        def __init__(self, c1, c2, k=5):
            super().__init__()
            Conv = YOLOv8_common.Conv
            c_ = c1 // 2
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c_ * 4, c2, 1, 1)
            self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        def forward(self, x):
            x = self.cv1(x)
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))