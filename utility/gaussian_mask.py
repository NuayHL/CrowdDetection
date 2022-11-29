import torch
import torch.nn as nn
import numpy as np
import math

class Gaussian_Mask_2D:
    def __init__(self, mu1, mu2, sigma1, sigma2, rho=0.0, max_one=False):
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho

        var1 = sigma1 * sigma1
        var2 = sigma2 * sigma2
        s1s2_2 = 2 * sigma2 * sigma1

        self.denominator = - 1 / (var1 * var2 * 2 * (1 - rho * rho))

        self.c_x_2 = var2 * self.denominator
        self.c_y_2 = var1 * self.denominator

        mu1_s2_2 = 2 * mu1 * var2
        mu2_s1_2 = 2 * mu2 * var1
        self.c_x_1 = (s1s2_2 * rho * mu2 - mu1_s2_2) * self.denominator
        self.c_y_1 = (s1s2_2 * rho * mu1 - mu2_s1_2) * self.denominator

        self.c_xy = - 2 * rho * sigma1 * sigma2 * self.denominator

        self.c_n = (sigma2 ** 2 * mu1 ** 2 + sigma1 ** 2 * mu2 ** 2 -
                    s1s2_2 * rho * mu1 * mu2) * self.denominator

        if max_one:
            self.c_pri = 1 / math.exp(self.c_xy * mu1 * mu2
                                      + self.c_x_2 * mu1 ** 2 + self.c_x_1 * mu1
                                      + self.c_y_2 * mu2 ** 2 + self.c_y_1 * mu2
                                      + self.c_n)
        else:
            self.c_pri = 1 / (2 * math.pi * sigma1 * sigma2 * math.sqrt(1 - self.rho * self.rho))


    def __call__(self, anchor_points):
        """anchor_points: The coordinates"""
        anchor_x = anchor_points[..., 0]
        anchor_y = anchor_points[..., 1]

        x_cal = self.c_x_2 * anchor_x * anchor_x + self.c_x_1 * anchor_x
        y_cal = self.c_y_2 * anchor_y * anchor_y + self.c_y_1 * anchor_y
        xy_cal = self.c_xy * anchor_x * anchor_y

        exp_cal = xy_cal + x_cal + y_cal + self.c_n

        if isinstance(anchor_points, np.ndarray):
            return self.c_pri * np.exp(exp_cal)
        elif isinstance(anchor_points, torch.Tensor):
            return self.c_pri * torch.exp(exp_cal)

def gen_gaussian_mask_gen(gts, sigma_ratio=(0.25, 0.4)):
    masks = list()
    x_r = sigma_ratio[0]
    y_r = sigma_ratio[1]
    for gt in gts:
        masks.append(Gaussian_Mask_2D(gt[0], gt[1], gt[2]*x_r, gt[3]*y_r, max_one=True))
    def multi_mask(x):
        return sum([mask(x) for mask in masks])
    return multi_mask

class Conv_Mask_2D:
    def __init__(self, input_channels=1, kernel_size=7, yita=0.5):
        kernel_x = np.arange(0, kernel_size, 1)
        kernel_y = np.arange(0, kernel_size, 1)
        kernel_x, kernel_y = np.meshgrid(kernel_x, kernel_y)
        kernel = torch.from_numpy(np.stack([kernel_x, kernel_y], axis=0))
        padding = int(kernel_size//2)
        dist, _ = torch.max(torch.abs(padding-kernel), dim=0, keepdim=True)

        weight = 1 - dist.float()/(padding + 1)

        self.weight = weight.unsqueeze(dim=0).tile(1, input_channels, 1, 1)
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv.weight = nn.Parameter(self.weight)
        self.yita = yita

    def __call__(self, x):
        self.conv.weight = nn.Parameter(self.weight.to(x.dtype).to(x.device))
        x = torch.clamp(self.conv(x), min=0.0, max=1.0)
        x = x * self.yita + 1 - self.yita
        return x

    def set_weight(self, weight):
        assert isinstance(weight, torch.Tensor) and weight.shape == self.weight.shape
        self.weight = weight

class Conv_Mask_2D_trainable(nn.Module):
    def __init__(self, kernel_size):
        super(Conv_Mask_2D_trainable, self).__init__()
        kernel_x = np.arange(0, kernel_size, 1)
        kernel_y = np.arange(0, kernel_size, 1)
        kernel_x, kernel_y = np.meshgrid(kernel_x, kernel_y)
        kernel = torch.from_numpy(np.stack([kernel_x, kernel_y], axis=0))
        padding = int(kernel_size//2)
        dist, _ = torch.max(torch.abs(padding-kernel), dim=0, keepdim=True)

        weight = 1 - dist.float()/(padding + 1)

        self.weight = weight.unsqueeze(dim=0)
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv.weight = nn.Parameter(weight.unsqueeze(dim=0))

    def forward(self, x):
        x = torch.clamp(self.conv(x), min=0.0, max=1.0)
        x = x * 0.5 + 0.5
        return x

class Conv_Mask_2D_trainable_soft(nn.Module):
    def __init__(self, kernel_size):
        super(Conv_Mask_2D_trainable_soft, self).__init__()
        kernel_x = np.arange(0, kernel_size, 1)
        kernel_y = np.arange(0, kernel_size, 1)
        kernel_x, kernel_y = np.meshgrid(kernel_x, kernel_y)
        kernel = torch.from_numpy(np.stack([kernel_x, kernel_y], axis=0))
        padding = int(kernel_size//2)
        dist, _ = torch.max(torch.abs(padding-kernel), dim=0, keepdim=True)

        weight = 1 - dist.float()/(padding + 1)

        self.weight = weight.unsqueeze(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv.weight = nn.Parameter(weight.unsqueeze(dim=0))
        self.soft_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float, requires_grad=True))

    def forward(self, x):
        x = self.sigmoid(self.conv(x))
        x = x * self.soft_weight + (1 - self.soft_weight)
        return x

if __name__ == '__main__':
    import cv2
    import os
    path = os.getcwd()
    os.chdir(os.path.join(path, '..'))

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')

    # conv_mask = Conv_Mask_2D_trainable_soft(7)

    a = torch.zeros((1, 2, 9, 9))

    a[:, :, 4, 4] = 1

    conv_mask = Conv_Mask_2D(7, 2)

    print(conv_mask(a))


    # from config import get_default_cfg
    #
    # cfg = get_default_cfg()
    #
    # from odcore.data.dataset import CocoDataset
    # dataset = CocoDataset('CrowdHuman/annotation_train_coco_style.json', 'CrowdHuman/Images_train', cfg.data, 'val')
    #
    # sample = dataset[2998]
    # img = sample['img']
    # gts = sample['anns']
    #
    # gen_mask = gen_gaussian_mask_gen(gts, sigma_ratio=(0.3, 0.4))
    #
    # anchorx = np.arange(0, cfg.data.input_width, 1)
    # anchory = np.arange(0, cfg.data.input_height, 1)
    # anchorx, anchory = np.meshgrid(anchorx, anchory)
    # anchors = np.stack((anchorx, anchory), axis=2)
    #
    # heatmap = np.expand_dims(gen_mask(anchors), 2)
    #
    # heatmap = heatmap.clip(min=0, max=1)
    #
    # heatmap = (heatmap * 255).astype(np.uint8)
    #
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:,:,::-1]
    #
    # finimg = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    #
    # fig, ax = plt.subplots()
    # ax.imshow(finimg)
    # ax.axis('off')
    # plt.show()
