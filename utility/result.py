import numpy as np
import torch

class Result():
    '''
    result format:
        N x 6: np.ndarray
        6: x1y1x2y2 target_score class_index
    '''
    def __init__(self, result, id, ori_shape, input_shape):
        '''input result: torch.Tensor'''
        self.result = result.detach().cpu().numpy()
        self.id = id
        self.input_shape = input_shape #(w, h)
        self.ori_shape = ori_shape
    def to_list_for_json(self, reconstruct_shape=True, letterboxinput=True):
        '''experiment_input_shape: indicating the input of network to restore the size'''
        fin_list = []
        if self.result.shape[0] == 0: return fin_list
        outresult = self.to_ori_label(reconstruct_shape, letterboxinput)
        # x1y1x2y2 to x1y1wh
        outresult[:, 2] -= outresult[:, 0]
        outresult[:, 3] -= outresult[:, 1]
        for dt in outresult:
            dt_dict = {}
            dt_dict['image_id'] = self.id
            dt_dict['category_id'] = int(dt[5].tolist())
            dt_dict['bbox'] = dt[:4].tolist()
            dt_dict['score'] = dt[4].tolist()
            fin_list.append(dt_dict)
        return fin_list

    def to_ori_label(self, reconstruct_shape=True, letterboxinput=True):
        '''x1y1x2y2 output, shape(w, h)'''
        if self.result.shape[0] == 0: return np.zeros((1,5))
        outresult = self.result.copy()
        if reconstruct_shape:
            if letterboxinput:
                rw = self.input_shape[0]/self.ori_shape[0]
                rh = self.input_shape[1]/self.ori_shape[1]
                if rh > rw:
                    r = rw
                    pad = (self.input_shape[1] - self.ori_shape[1] * r) / 2
                    outresult[:, 1] -= pad
                    outresult[:, 3] -= pad
                else:
                    r = rh
                    pad = (self.input_shape[0] - self.ori_shape[0] * r) / 2
                    outresult[:, 0] -= pad
                    outresult[:, 2] -= pad
                outresult[:, :4] /= r
            else:
                outresult[:, 0] *= self.ori_shape[0] / float(self.input_shape[0])
                outresult[:, 2] *= self.ori_shape[0] / float(self.input_shape[0])
                outresult[:, 1] *= self.ori_shape[1] / float(self.input_shape[1])
                outresult[:, 3] *= self.ori_shape[1] / float(self.input_shape[1])
        return outresult

    @staticmethod
    def result_parse_for_json(results):
        result_list = []
        for result in results:
            result_list.extend(result.to_list_for_json())
        return result_list