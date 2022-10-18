import numpy as np
import math
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
from utility.anchors import result_parse
from odcore.utils.visualization import stack_img, generate_hot_bar


class HotMapHooker:
    data = dict()

    def __init__(self, config):
        HotMapHooker.config = config

    def get_head_hot_map_hooker(self):
        return self._head_hot_map_hooker

    def get_neck_hot_map_hooker(self):
        return self._neck_hot_map_hooker

    @staticmethod
    def _head_hot_map_hooker(module, input, output):
        cfg = HotMapHooker.config
        sigmoid = nn.Sigmoid()
        hotmap = sigmoid(output[:, 4, :])
        assert len(hotmap.shape) == 2, "please using single batch input"
        hot_map = hotmap.t().detach().cpu()
        HotMapHooker.data['head_hot_map'] = hot_map
        hot_map_list = result_parse(cfg, hot_map, restore_size=True)

        sum_result = []
        fig, ax = plt.subplots(3, 4)
        for id, level in enumerate(hot_map_list):
            for il, fm in enumerate(level):
                fm = fm.numpy()
                sum_result.append(fm)
                ax[id][il].imshow(fm)
                ax[id][il].axis('off')
        plt.show()

        fpnlevels = len(cfg.model.fpnlevels)
        anchor_per_grid = len(cfg.model.anchor_ratios[0]) if cfg.model.use_anchor else 1

        sum_result = stack_img(sum_result, (fpnlevels, anchor_per_grid))
        bar = generate_hot_bar(1.0, 0.0, sum_result.shape[0])
        sum_result = np.concatenate([sum_result, bar], axis=1)
        HotMapHooker.data['head_hot_map_img'] = sum_result
        fig, ax = plt.subplots()
        ax.imshow(sum_result)
        ax.axis('off')
        plt.show()

    @staticmethod
    def _neck_hot_map_hooker(module, input, output):
        assert input[0].shape[0] == 1, "please using single batch input"
        in_layer = [layer.squeeze(0).permute((1, 2, 0)).detach().cpu().numpy() for layer in input]
        out_layer = [layer.squeeze(0).permute((1, 2, 0)).detach().cpu().numpy() for layer in output]

        in_layer = [HotMapHooker.return_hot_map_from_feature(level, add_bar=False) for level in in_layer]

        fig, ax = plt.subplots()
        ax.imshow(in_layer[2])
        ax.axis('off')
        plt.show()

    @staticmethod
    def return_hot_map_from_feature(block: np.ndarray, add_bar=True):
        width, height, channel = block.shape
        stack_number = int(math.sqrt(float(channel)))

        channel_min = block.min(axis=(0, 1))
        channel_range = block.max(axis=(0, 1)) - channel_min

        channel_list_image = [block[:, :, i:(i+1)] for i in range(channel)]
        channel_list_image = [(channel-min_p)/range_p for channel, min_p, range_p in zip(channel_list_image,
                                                                                         channel_min,
                                                                                         channel_range)]
        try:
            channel_image = stack_img(channel_list_image, shape=(stack_number, stack_number))
        except:
            return None

        if add_bar:
            bar = generate_hot_bar(1.0, 0.0, channel_image.shape[0])
            channel_image = np.concatenate([channel_image, bar], axis=1)
        return channel_image

