import numpy as np
import torch

class Result():
    '''
    result format:
        N x 6: np.ndarray
        6: x1y1x2y2 target_score class_index
    '''
    def __init__(self, result, id, shape):
        '''input result: torch.Tensor'''
        self.result = result.detach().cpu().numpy()
        self.id = id
        self.ori_shape = shape #(w, h)
    def to_list_for_json(self, experiment_input_shape=None, letterboxinput=True):
        '''experiment_input_shape: indicating the input of network to restore the size'''
        fin_list = []
        if self.result.shape[0] == 0: return fin_list
        outresult = self.to_ori_label(experiment_input_shape, letterboxinput)
        for dt in outresult:
            dt_dict = {}
            dt_dict['image_id'] = self.id
            dt_dict['category_id'] = dt[5]
            dt_dict['bbox'] = dt[:4]
            dt_dict['score'] = dt[4]
            fin_list.append(dt_dict)
        return fin_list
    def to_ori_label(self, experiment_input_shape=None, letterboxinput=True):
        if self.result.shape[0] == 0: return np.zeros((1,5))
        outresult = self.result.copy()
        if experiment_input_shape:
            if letterboxinput:
                rw = experiment_input_shape[0]/self.ori_shape[0]
                rh = experiment_input_shape[1]/self.ori_shape[1]
                if rh > rw:
                    r = rw
                    pad = (experiment_input_shape[1] - self.ori_shape[1] * r) / 2
                    outresult[:, 1] -= pad
                    outresult[:, 3] -= pad
                else:
                    r = rh
                    pad = (experiment_input_shape[0] - self.ori_shape[0] * r) / 2
                    outresult[:, 0] -= pad
                    outresult[:, 2] -= pad
                outresult[:, :4] /= r
            else:
                outresult[:, 0] *= self.ori_shape[0] / float(experiment_input_shape[0])
                outresult[:, 2] *= self.ori_shape[0] / float(experiment_input_shape[0])
                outresult[:, 1] *= self.ori_shape[1] / float(experiment_input_shape[1])
                outresult[:, 3] *= self.ori_shape[1] / float(experiment_input_shape[1])
        return outresult