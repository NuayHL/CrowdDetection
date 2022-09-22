import torch
import torch.nn as nn
import torch.cuda.amp as amp

from utility.assign import get_assign_method
from utility.anchors import generateAnchors, result_parse, Anchor
from utility.loss import GeneralLoss_fix, updata_loss_dict
from utility.nms import non_max_suppression
from utility.result import Result

from utility.nms_copy import cpu_nms

# model.set(args, device)
# model.coco_parse_result(results) results: List of prediction

class YoloX(nn.Module):
    '''4 + 1 + classes'''
    def __init__(self, config, backbone, neck, head):
        super(YoloX, self).__init__()
        self.config = config
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.sigmoid = nn.Sigmoid()
        self.input_shape = (self.config.data.input_width, self.config.data.input_height)

    def forward(self, sample):
        if self.training:
            return self.training_loss(sample)
        else:
            return self.inferencing(sample)

    def core(self,input):
        p3, p4, p5 = self.backbone(input)
        p3, p4, p5 = self.neck(p3, p4, p5)
        dt = self.head(p3, p4, p5)
        return dt

    def set(self, args, device):
        self.device = device
        self.anchors_per_grid = len(self.config.model.anchor_ratios) * len(self.config.model.anchor_scales)
        self.assignment = get_assign_method(self.config, device)
        self.assign_type = self.config.model.assignment_type.lower()
        if self.config.model.use_anchor:
            self.anchs = torch.from_numpy(generateAnchors(self.config, singleBatch=True)).float().to(device)
        else:
            raise NotImplementedError('Current Yolo do not support anchor free')
        self.num_of_proposal = self.anchs.shape[0]
        # assert self.config.data.ignored_input is True, "Please set the config.data.ignored_input as True"

        self.use_l1 = True if 'l1' in self.config.loss.reg_type else False

        assert True in ['iou' in loss for loss in self.config.loss.reg_type], 'Please adding iou-type loss'
        self.loss = GeneralLoss_fix(self.config, device, reduction='mean')

    def training_loss(self,sample):
        dt = self.core(sample['imgs']) #[B, 4+1+cls, n_anchors_all]
        num_of_class = dt.shape[1]-5
        cls_dt = dt[:, 5:, :].clone()  #[B, cls, n_anchors_all]
        obj_dt = dt[:, 4, :].clone()   #[B, n_anchors_all]
        ori_reg_dt = dt[:, :4, :]      #[B, 4, n_anchors_all]
        shift_dt = self.get_shift_bbox(ori_reg_dt)

        # with amp.autocast(enabled=False):
        with torch.no_grad():
            if self.assign_type == 'simota':
                assign_result, gt, weight = self.assignment.assign(sample['annss'],
                                                                   torch.cat([shift_dt, obj_dt.unsqueeze(1), cls_dt], dim=1))
            else:
                assign_result, gt, weight = self.assignment.assign(sample['annss'])

        pos_mask = []

        cls_gt = []
        obj_gt = []
        shift_gt = []
        if self.use_l1:
            l1_gt = []

        for ib in range(len(gt)):
            assign_result_ib, gt_ib = assign_result[ib], gt[ib]
            pos_mask_ib = torch.gt(assign_result_ib, 0.5)
            pos_mask.append(pos_mask_ib)

            label_pos_generate = (assign_result_ib[pos_mask_ib] - 1).long()

            shift_gt_ib = gt_ib[label_pos_generate, :4].t()
            shift_gt.append(shift_gt_ib)
            if self.use_l1:
                l1_gt_ib = gt_ib[label_pos_generate, :4]
                l1_gt_ib = self.get_l1_target(l1_gt_ib, pos_mask_ib)
                l1_gt.append(l1_gt_ib)
            cls_gt_ib = torch.zeros((self.config.data.numofclasses, self.num_of_proposal),
                                    dtype=torch.float32).to(self.device)
            cls_gt_ib[gt_ib[label_pos_generate,4].long(), pos_mask_ib] = 1
            cls_gt_ib = cls_gt_ib[:, pos_mask_ib]
            cls_gt.append(cls_gt_ib)

            obj_gt_ib = assign_result_ib.clamp(0, 1)
            if weight:
                obj_gt_ib[pos_mask_ib] *= weight[ib]
            obj_gt.append(obj_gt_ib)

        pos_mask = torch.cat(pos_mask, dim=0)
        dt = {}
        gt = {}
        dt['obj'] = obj_dt.view(-1)
        gt['obj'] = torch.cat(obj_gt,dim=0)
        dt['cls'] = cls_dt.permute(1,0,2).reshape(num_of_class,-1)[:, pos_mask]
        gt['cls'] = torch.cat(cls_gt,dim=1)
        dt['iou'] = shift_dt.permute(1,0,2).reshape(4,-1)[:, pos_mask]
        gt['iou'] = torch.cat(shift_gt, dim=1)
        if self.use_l1:
            dt['l1'] = ori_reg_dt.permute(1,0,2).reshape(4,-1)[:, pos_mask]
            gt['l1'] = torch.cat(l1_gt, dim=1)

        return self.loss(dt, gt)

    def inferencing(self, sample):
        dt = self.core(sample['imgs'])
        anchors = self.anchs.t()
        anchors = torch.tile(anchors, (dt.shape[0], 1, 1))

        # restore the predicting bboxes via pre-defined anchors
        if self.config.model.use_anchor:
            dt[:, 2:4, :] = anchors[:, 2:, :] * torch.exp(dt[:, 2:4, :])
            dt[:, :2, :] = anchors[:, :2, :] + dt[:, :2, :] * anchors[:, 2:, :]

        dt[:, 4:, :] = self.sigmoid(dt[:, 4:, :])

        dt = torch.permute(dt, (0,2,1))
        if self.config.inference.nms_type == 'nms':
            result_list = non_max_suppression(dt, conf_thres=self.config.inference.obj_thres,
                                              iou_thres=self.config.inference.iou_thres)
        else:
            raise NotImplementedError
        fin_result = []
        for result, id, ori_shape in zip(result_list, sample['ids'],sample['shapes']):
            fin_result.append(Result(result, id, ori_shape, self.input_shape))
        return fin_result

    def get_shift_bbox(self, ori_box:torch.Tensor): # return xywh Bbox
        if self.config.model.use_anchor:
            shift_box = ori_box.clone().to(torch.float32)
            anchors = torch.tile(self.anchs.t(), (shift_box.shape[0], 1, 1))
            shift_box[:, 2:] = anchors[:, 2:] * torch.exp(ori_box[:, 2:4].clamp(max=25))
            shift_box[:, :2] = anchors[:, :2] + ori_box[:, :2] * anchors[:, 2:]
        else:
            raise NotImplementedError
        return shift_box

    def get_l1_target(self, gt_pos_mask_generate, pos_mask):
        if self.config.model.use_anchor:
            l1_gt_ib = gt_pos_mask_generate
            l1_gt_ib[:, 2:] = torch.log(l1_gt_ib[:, 2:] / self.anchs[pos_mask, 2:])
            l1_gt_ib[:, :2] = (l1_gt_ib[:, :2] - self.anchs[pos_mask, :2]) / self.anchs[pos_mask, 2:]
        else:
            raise NotImplementedError
        return l1_gt_ib.t()

    def _test_hot_map(self, sample):
        dt = self.core(sample['imgs'])
        hot_map = self.sigmoid(dt[:, 4, :])
        assert len(hot_map.shape) == 2, "please using single batch input"
        hot_map = hot_map.t().detach().cpu()
        hot_map_list = result_parse(self.config, hot_map, restore_size=True)
        return hot_map_list

    @staticmethod
    def coco_parse_result(results):
        return Result.result_parse_for_json(results)

    def _debug_to_file(self, *args,**kwargs):
        with open('debug.txt', 'a') as f:
            print(*args,**kwargs,file=f)


