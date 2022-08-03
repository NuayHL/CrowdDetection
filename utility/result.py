import torch

class Result():
    '''
    result format:
        N x 6: np.ndarray
        6: x1y1x2y2 target_score class_index
    '''
    def __init__(self, result):
        if isinstance(result, torch.Tensor): result = result.numpy()
        self.result = result
        self.len = result.shape[0]