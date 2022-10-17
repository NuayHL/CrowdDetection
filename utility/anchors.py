import numpy as np

class Anchor():
    def __init__(self, config):
        self.config = config
        self.fpnlevels = config.model.fpnlevels
        self.basesize = [2 ** (x + 2) for x in self.fpnlevels]
        self.ratios = config.model.anchor_ratios
        self.scales = config.model.anchor_scales

    def gen_Bbox(self, singleBatch=False):
        # formate = xywh
        allAnchors = np.zeros((0, 4)).astype(np.float32)
        assert len(self.ratios) == len(self.fpnlevels)
        assert len(self.scales) == len(self.fpnlevels)
        self.anchors_per_grid = len(self.ratios[0])
        for grid_indi in self.ratios + self.scales:
            assert len(grid_indi) == self.anchors_per_grid

        for idx, p in enumerate(self.fpnlevels):
            stride = [2 ** p, 2 ** p]
            xgrid = np.arange(0, self.config.data.input_width, stride[0]) + stride[0] / 2.0
            ygrid = np.arange(0, self.config.data.input_height, stride[1]) + stride[1] / 2.0
            xgrid, ygrid = np.meshgrid(xgrid, ygrid)
            anchors = np.vstack((xgrid.ravel(), ygrid.ravel()))
            lenAnchors = anchors.shape[1]
            anchors = np.tile(anchors, (2, self.anchors_per_grid)).T
            start = 0
            # for ratio in self.ratios:
            #     for scale in self.scales:
            #         print(ratio, scale)
            #         anchors[start:start + lenAnchors, 2] = self.basesize[idx] * scale
            #         anchors[start:start + lenAnchors, 3] = self.basesize[idx] * scale * ratio
            #         start += lenAnchors
            for ratio, scale in zip(self.ratios[idx], self.scales[idx]):
                anchors[start:start + lenAnchors, 2] = self.basesize[idx] * scale
                anchors[start:start + lenAnchors, 3] = self.basesize[idx] * scale * ratio
                start += lenAnchors

            allAnchors = np.append(allAnchors, anchors, axis=0)

        # singleBatch return total_anchor_number X 4
        if singleBatch: return allAnchors

        # batchedAnchor return Batchsize X total_anchor_number X 4
        allAnchors = np.tile(allAnchors, (self.config.training.batch_size, 1, 1))
        return allAnchors

    def gen_points(self,singleBatch=False):
        # formate = xy
        allPoints = np.zeros((0, 2)).astype(np.float32)
        for idx, p in enumerate(self.fpnlevels):
            stride = [2 ** p, 2 ** p]
            xgrid = np.arange(0, self.config.data.input_width, stride[0]) + stride[0] / 2.0
            ygrid = np.arange(0, self.config.data.input_height, stride[1]) + stride[1] / 2.0
            xgrid, ygrid = np.meshgrid(xgrid, ygrid)
            points = np.vstack((xgrid.ravel(), ygrid.ravel())).T
            allPoints = np.append(allPoints, points, axis=0)

        # singleBatch return total_anchor_number X 4
        if singleBatch: return allPoints

        # batchedAnchor return Batchsize X total_anchor_number X 4
        allPoints = np.tile(allPoints, (self.config.training.batch_size, 1, 1))
        return allPoints

    def gen_stride(self, singleBatch=True):
        num_in_each_level = np.ones(len(self.fpnlevels), dtype=np.int)
        for id, i in enumerate(self.fpnlevels):
            temp = self.config.data.input_width * self.config.data.input_height / (2 ** (2*i))
            if self.config.model.use_anchor:
                assert len(self.ratios) == len(self.fpnlevels)
                assert len(self.scales) == len(self.fpnlevels)
                anchors_per_grid = len(self.ratios[0])
                for grid_indi in self.ratios + self.scales:
                    assert len(grid_indi) == anchors_per_grid
                temp *= anchors_per_grid
            num_in_each_level[id] *= temp
        stride = np.ones(num_in_each_level.sum())
        start_index = 0
        for i, num in zip(self.fpnlevels, num_in_each_level):
            stride[start_index: start_index+num] *= 2**i
            start_index += num
        if singleBatch: return stride
        return np.tile(stride, (self.config.training.batch_size, 1))

    def get_anchors_per_grid(self):
        return len(self.ratios[0]) if self.config.model.use_anchor else 1

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

