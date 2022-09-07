import torch
import torch.nn as nn

from utility.assign import AnchorAssign
from utility.anchors import generateAnchors
from utility.loss import GeneralLoss, updata_loss_dict
from utility.nms import non_max_suppression
from utility.result import Result

# model.set(args, device)
# model.coco_parse_result(results) results: List of prediction

class Yolov3(nn.Module):
    '''4 + 1 + classes'''
    def __init__(self, config, backbone, neck, head):
        super(Yolov3, self).__init__()
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
        self.assignment = AnchorAssign(self.config, device)
        self.loss = GeneralLoss(self.config, device)
        if self.config.model.use_anchor:
            self.anchs = torch.from_numpy(generateAnchors(self.config, singleBatch=True)).float().to(device)
            self.num_of_proposal = self.anchs.shape[0]
        else:
            raise NotImplementedError('Current Yolo do not support anchor free')
        assert self.config.data.ignored_input is True, "Please set the config.data.ignored_input as True"

        self.loss_order = []
        self.use_l1 = False
        for type in self.config.loss.reg_type:
            if 'l1' in type:
                self.loss_order.append('l1')
                self.use_l1 = True
            if 'iou' in type: self.loss_order.append('iou')
        assert 'iou' in self.loss_order, 'Please adding iou-type loss'

    def training_loss(self,sample):
        dt = self.core(sample['imgs'])
        cls_dt = dt[:, 5:, :]
        obj_dt = dt[:, 4, :]
        ori_reg_dt = dt[:, :4, :]
        shift_dt = self.get_shift_bbox(ori_reg_dt)

        assign_result, gt = self.assignment.assign(sample['annss'])

        pos_mask = []

        cls_gt = []
        obj_gt = []
        shift_gt = []
        if self.use_l1:
            l1_gt = []

        # first construct simOTA!!
        for ib in range(len(gt)):
            assign_result_ib, gt_ib = assign_result[ib], gt[ib]
            pos_mask_ib = torch.gt(assign_result_ib, 0.5)
            pos_mask.append(pos_mask_ib)

            label_pos_generate = (assign_result_ib[pos_mask] - 1).long()

            shift_gt_ib = gt_ib[label_pos_generate, :4].t()
            shift_gt.append(shift_gt_ib)
            if self.use_l1:
                l1_gt_ib = gt_ib[label_pos_generate, :4]
                l1_gt_ib[:, 2:] = torch.log(l1_gt_ib[:, 2:] / self.anchs[pos_mask, 2:])
                l1_gt_ib[:, :2] = (l1_gt_ib[:, :2] - self.anchs[pos_mask, :2]) / self.anchs[pos_mask, 2:]
                l1_gt_ib = l1_gt_ib.t()
                l1_gt.append(l1_gt_ib)
            cls_gt_ib = torch.zeros((self.config.data.numofclasses, self.num_of_proposal),
                                    dtype=torch.float32).to(self.device)
            cls_gt_ib[gt_ib[label_pos_generate,4].long(), pos_mask] = 1
            cls_gt_ib = cls_gt_ib[:, pos_mask]
            cls_gt.append(cls_gt_ib)

            obj_gt_ib = assign_result_ib.clamp(0, 1)
            obj_gt.append(obj_gt_ib)

        pos_mask = torch.
        ## view bbox ?

        num_pos_samples = max(num_pos_samples.item(),1)
        for key in fin_loss_dict:
            fin_loss_dict[key] /= num_pos_samples
        return fin_loss/num_pos_samples, fin_loss_dict

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
            fin_result.append(Result(result, id, ori_shape,self.input_shape))
        return fin_result

    def get_shift_bbox(self, ori_box:torch.Tensor):
        if self.config.model.use_anchor:
            shift_box = ori_box.clone()
            anchors = torch.tile(self.anchs.t(), (shift_box.shape[0], 1, 1))
            shift_box[:, 2:] = anchors[:, 2:] * torch.exp(ori_box[:, 2:4].clamp(max=50))
            shift_box[:, :2] = anchors[:, :2] + ori_box[:, :2] * anchors[:, 2:]
        else:
            raise NotImplementedError

        return shift_box

    @staticmethod
    def coco_parse_result(results):
        return Result.result_parse_for_json(results)

    def _debug_to_file(self, *args,**kwargs):
        with open('debug.txt', 'a') as f:
            print(*args,**kwargs,file=f)


