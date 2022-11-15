import numpy as np
from copy import deepcopy

class Anchor():
    def __init__(self, config):
        self.config = config
        self.using_anchor = config.model.use_anchor
        self.fpnlevels = config.model.fpnlevels
        self.basesize = [2 ** (x + 2) for x in self.fpnlevels]
        self.ratios = config.model.anchor_ratios
        self.scales = config.model.anchor_scales
        self.free_scale = config.model.stride_scale

        self.using_mip = self.config.model.assignment_type.lower() in ['mip', 'MIP']
        self.mip_k = None if not self.using_mip else int(self.config.model.assignment_extra[0]['k'])

    def gen_Bbox(self, singleBatch=False, act_mip_mode_if_use=True):
        # formate = xywh
        allAnchors = np.zeros((0, 4)).astype(np.float32)
        assert len(self.ratios) == len(self.fpnlevels)
        assert len(self.scales) == len(self.fpnlevels)
        anchors_per_grid = len(self.ratios[0])
        for grid_indi in self.ratios + self.scales:
            assert len(grid_indi) == anchors_per_grid

        ratios = deepcopy(self.ratios)
        scales = deepcopy(self.scales)

        # If use MIP..
        if self.using_mip and act_mip_mode_if_use:
            anchors_per_grid *= self.mip_k
            ratios = [self.mip_repeator(ratio, self.mip_k) for ratio in ratios]
            scales = [self.mip_repeator(scale, self.mip_k) for scale in scales]

        for idx, p in enumerate(self.fpnlevels):
            stride = [2 ** p, 2 ** p]
            xgrid = np.arange(0, self.config.data.input_width, stride[0]) + stride[0] / 2.0
            ygrid = np.arange(0, self.config.data.input_height, stride[1]) + stride[1] / 2.0
            xgrid, ygrid = np.meshgrid(xgrid, ygrid)
            anchors = np.vstack((xgrid.ravel(), ygrid.ravel()))
            lenAnchors = anchors.shape[1]
            anchors = np.tile(anchors, (2, anchors_per_grid)).T
            start = 0
            for ratio, scale in zip(ratios[idx], scales[idx]):
                anchors[start:start + lenAnchors, 2] = self.basesize[idx] * scale
                anchors[start:start + lenAnchors, 3] = self.basesize[idx] * scale * ratio
                start += lenAnchors

            allAnchors = np.append(allAnchors, anchors, axis=0)

        # singleBatch return total_anchor_number X 4
        if singleBatch:
            return allAnchors

        # batchedAnchor return Batchsize X total_anchor_number X 4
        allAnchors = np.tile(allAnchors, (self.config.training.batch_size, 1, 1))
        return allAnchors

    def gen_points(self, singleBatch=False, act_mip_mode_if_use=True):
        # formate = xy
        allPoints = np.zeros((0, 2)).astype(np.float32)
        for idx, p in enumerate(self.fpnlevels):
            stride = [2 ** p, 2 ** p]
            xgrid = np.arange(0, self.config.data.input_width, stride[0]) + stride[0] / 2.0
            ygrid = np.arange(0, self.config.data.input_height, stride[1]) + stride[1] / 2.0
            xgrid, ygrid = np.meshgrid(xgrid, ygrid)
            points = np.vstack((xgrid.ravel(), ygrid.ravel())).T
            allPoints = np.append(allPoints, points, axis=0)

            # If use MIP..
            if self.using_mip and act_mip_mode_if_use:
                for _ in range(self.mip_k-1):
                    allPoints = np.append(allPoints, points, axis=0)

        # singleBatch return total_anchor_number X 4
        if singleBatch:
            return allPoints

        # batchedAnchor return Batchsize X total_anchor_number X 4
        allPoints = np.tile(allPoints, (self.config.training.batch_size, 1, 1))
        return allPoints

    def gen_stride(self, singleBatch=True):
        num_in_each_level = np.ones(len(self.fpnlevels), dtype=np.int)
        size_in_each_level = np.ones(len(self.fpnlevels), dtype=np.int)
        anchors_per_grid = self.get_anchors_per_grid()
        for id, i in enumerate(self.fpnlevels):
            temp = self.config.data.input_width * self.config.data.input_height / (2 ** (2*i))
            size_in_each_level[id] *= temp
            if self.using_anchor:
                assert len(self.ratios) == len(self.fpnlevels)
                assert len(self.scales) == len(self.fpnlevels)
                for grid_indi in self.ratios + self.scales:
                    assert len(grid_indi) == anchors_per_grid
                temp *= anchors_per_grid
            num_in_each_level[id] *= temp
        stride = np.ones(num_in_each_level.sum())
        start_index = 0
        if not self.using_anchor:
            for i, num in zip(self.fpnlevels, num_in_each_level):
                stride[start_index: start_index+num] *= (2**i)*self.free_scale
                start_index += num
        else:
            for i, scales, size in zip(self.fpnlevels, self.scales, size_in_each_level):
                for scale in scales:
                    stride[start_index: start_index+size] *= (2**i)*scale
                    start_index += size
        if singleBatch:
            return stride
        return np.tile(stride, (self.config.training.batch_size, 1))

    def gen_ratio(self, singleBatch=True):
        if not self.using_anchor:
            'Need adding warning'
            return None
        num_in_each_level = np.ones(len(self.fpnlevels), dtype=np.int)
        size_in_each_level = []
        anchors_per_grid = self.get_anchors_per_grid()
        for id, i in enumerate(self.fpnlevels):
            temp = self.config.data.input_width * self.config.data.input_height / (2 ** (2*i))
            size_in_each_level.append(int(temp))
            assert len(self.ratios) == len(self.fpnlevels)
            assert len(self.scales) == len(self.fpnlevels)
            for grid_indi in self.ratios + self.scales:
                assert len(grid_indi) == anchors_per_grid
            temp *= anchors_per_grid
            num_in_each_level[id] *= temp
        fin_ratio = np.ones(num_in_each_level.sum())
        start_index = 0
        for i, ratios, size in zip(self.fpnlevels, self.ratios, size_in_each_level):
            for ratio in ratios:
                fin_ratio[start_index: start_index+size] *= ratio
                start_index += size
        if singleBatch:
            return fin_ratio
        return np.tile(fin_ratio, (self.config.training.batch_size, 1))

    def get_anchors_per_grid(self):
        return len(self.ratios[0]) if self.using_anchor else 1

    def get_num_in_each_level(self):
        num_list = []
        for level in self.fpnlevels:
            giw = int(self.config.data.input_width / (2 ** level))
            gih = int(self.config.data.input_height / (2 ** level))
            num_list.append(giw * gih * self.get_anchors_per_grid())
        return num_list

    def gen_block_indicator(self):
        num_in_each_block = []
        for level in self.fpnlevels:
            num_in_this_block = self.config.data.input_width * self.config.data.input_height / (2 ** (2 * level))
            for i in range(self.get_anchors_per_grid()):
                num_in_each_block.append(num_in_this_block)
        num_in_each_block = np.array(num_in_each_block, dtype=np.int)
        block_indicator = np.zeros((num_in_each_block.sum(), 1))
        starts = 0
        for i, num_in_block in enumerate(num_in_each_block):
            block_indicator[starts: starts+num_in_block, 0] = i
            starts += num_in_block
        return block_indicator

    @staticmethod
    def mip_repeator(input_list: list, k: int):
        return_list = list()
        for item in input_list:
            return_list += [item] * k
        return return_list

def generateAnchors(config, basesize=None, fpnlevels=None, ratios=None, scales=None, singleBatch=False):
    '''
    return: batch_size X total_anchor_numbers X 4
    anchor box type: xywh
    singleBatch: set True if only need batchsize at 1
    '''
    if fpnlevels == None:
        fpnlevels = config.model.fpnlevels
    if basesize == None:
        basesize = [2**(x+2) for x in fpnlevels]
    if ratios == None:
        # ratios = h/w
        ratios = config.model.anchor_ratios
    if scales == None:
        scales = config.model.anchor_scales

    assert len(fpnlevels) == len(basesize)

    # formate = xywh
    allAnchors = np.zeros((0,4)).astype(np.float32)
    for idx, p in enumerate(fpnlevels):
        stride = [2**p, 2**p]
        xgrid = np.arange(0, config.data.input_width, stride[0]) + stride[0] / 2.0
        ygrid = np.arange(0, config.data.input_height, stride[1]) + stride[1] / 2.0
        xgrid, ygrid = np.meshgrid(xgrid, ygrid)
        anchors = np.vstack((xgrid.ravel(),ygrid.ravel()))
        lenAnchors = anchors.shape[1]
        anchors = np.tile(anchors,(2,len(ratios)*len(scales))).T
        start = 0
        for ratio in ratios:
            for scale in scales:
                anchors[start:start+lenAnchors, 2] = basesize[idx] * scale
                anchors[start:start+lenAnchors, 3] = basesize[idx] * scale * ratio
                start += lenAnchors
        allAnchors = np.append(allAnchors,anchors,axis=0)

    # singleBatch return total_anchor_number X 4
    if singleBatch: return allAnchors

    # batchedAnchor return Batchsize X total_anchor_number X 4
    allAnchors = np.tile(allAnchors, (config.training.batch_size, 1, 1))
    return allAnchors

def anchors_parse(config, anchors, fplevel=None,ratios=None,scales=None):
    anchors[:, 0] += float(config.data.input_width/2)
    anchors[:, 1] += float(config.data.input_height/2)
    if fplevel is None:
        fplevel = config.model.fpnlevels
    if scales is None:
        scales = config.model.anchor_scales
    if ratios is None:
        ratios = config.model.anchor_ratios
    anchors_per_grid = len(scales) * len(ratios)
    width = config.data.input_width
    height = config.data.input_height
    begin_level = 0
    parsed_anch = []
    for i in fplevel:
        ilevel_anch = []
        i_w = width/(2**i)
        i_h = height/(2**i)
        for rs in range(anchors_per_grid):
            ilevel_anch.append(anchors[int(begin_level+rs*i_h*i_w):int(begin_level+(rs+1)*i_h*i_w),:])
        begin_level += anchors_per_grid*i_h*i_w
        parsed_anch.append(ilevel_anch)
    return parsed_anch

def result_parse(config, dt_liked, anchors_per_grid=None, restore_size=False):
    fplevel = config.model.fpnlevels
    if not anchors_per_grid:
        anchors_per_grid = len(config.model.anchor_scales[0]) if config.model.use_anchor else 1
    width = config.data.input_width
    height = config.data.input_height
    begin_level = 0
    parsed_anch = []
    for i in fplevel:
        ilevel_anch = []
        i_w = width/(2**i)
        i_h = height/(2**i)
        for rs in range(anchors_per_grid):
            temp = dt_liked[int(begin_level+rs*i_h*i_w):int(begin_level+(rs+1)*i_h*i_w), :]
            if restore_size:
                temp = temp.reshape(int(i_h), int(i_w), -1)
            ilevel_anch.append(temp)
        begin_level += anchors_per_grid*i_h*i_w
        parsed_anch.append(ilevel_anch)
    return parsed_anch

