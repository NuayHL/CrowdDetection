import numpy as np

def generateAnchors(config, fpnlevels=None, basesize=None, ratios=None, scales=None, singleBatch=False, ):
    '''
    return: batch_size X total_anchor_numbers X 4
    anchor box type: x1y1x2y2
    singleBatch: set True if only need batchsize at 1
    '''
    if fpnlevels == None:
        fpnlevels = config.anchorLevels
    if basesize == None:
        basesize = [2**(x+2) for x in fpnlevels]
    if ratios == None:
        # ratios = h/w
        ratios = config.anchorRatio
    if scales == None:
        scales = config.anchorScales

    # formate = x1y1x2y2
    allAnchors = np.zeros((0,4)).astype(np.float32)
    for idx, p in enumerate(fpnlevels):
        stride = [2**p, 2**p]
        xgrid = np.arange(0, config.input_width, stride[0]) + stride[0] / 2.0
        ygrid = np.arange(0, config.input_height, stride[1]) + stride[1] / 2.0
        xgrid, ygrid = np.meshgrid(xgrid, ygrid)
        anchors = np.vstack((xgrid.ravel(),ygrid.ravel()))
        lenAnchors = anchors.shape[1]
        anchors = np.tile(anchors,(2,len(ratios)*len(scales))).T
        start = 0
        for ratio in ratios:
            for scale in scales:
                anchors[start:start+lenAnchors, 0] -= basesize[idx] * scale / 2.0
                anchors[start:start+lenAnchors, 1] -= basesize[idx] * scale * ratio / 2.0
                anchors[start:start+lenAnchors, 2] += basesize[idx] * scale / 2.0
                anchors[start:start+lenAnchors, 3] += basesize[idx] * scale * ratio / 2.0
                start += lenAnchors
        allAnchors = np.append(allAnchors,anchors,axis=0)

    # singleBatch return total_anchor_number X 4
    if singleBatch: return allAnchors

    # batchedAnchor return Batchsize X total_anchor_number X 4
    allAnchors = np.tile(allAnchors, (config.batch_size, 1, 1))

    return allAnchors

def anchors_parse(config,fplevel=None,ratios=None,scales=None,anchors=generateAnchors(singleBatch=True)):
    anchors[:, 0] += config.input_width/2
    anchors[:, 2] += config.input_width/2
    anchors[:, 1] += config.input_height/2
    anchors[:, 3] += config.input_height/2
    if fplevel is None:
        fplevel = config.anchorLevels
    if scales is None:
        scales = config.anchorScales
    if ratios is None:
        ratios = config.anchorRatio
    anchors_per_grid = len(scales) * len(ratios)
    width = config.input_width
    height = config.input_height
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






