import torch
import torch.nn as nn

from utility.result import Result
from utility.anchors import result_parse


class BaseODModel(nn.Module):
    def set(self, args, device):
        raise NotImplementedError('please create function to receive args and device')

    def forward(self, sample):
        if self.training:
            return self.training_loss(sample)
        else:
            return self.inferencing(sample)

    def training_loss(self, sample):
        """strictly return (total_loss, loss_dict)"""
        raise NotImplementedError('Please create training loss function')

    def inferencing(self, sample):
        """related to your flexible design"""
        raise NotImplementedError('Please create inferencing function')

    @staticmethod
    def coco_parse_result(results):
        """
        Input: List: list[result]
        Output: List: list[{coco_pred1},{coco_pred2}...]
        """
        return Result.result_parse_for_json(results)

    def _debug_to_file(self, *args, **kwargs):
        """debug tools"""
        with open('debug.txt', 'a') as f:
            print(*args, **kwargs, file=f)

    def _test_hot_map(self, sample):
        dt = self.core(sample['imgs'])
        hot_map = self.sigmoid(dt[:, 4, :])
        assert len(hot_map.shape) == 2, "please using single batch input"
        hot_map = hot_map.t().detach().cpu()
        hot_map_list = result_parse(self.config, hot_map, restore_size=True)
        return hot_map_list