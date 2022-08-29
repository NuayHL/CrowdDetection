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
        p3, p4, p5 = self.head(p3, p4, p5)
        dt = self._result_parse((p3, p4, p5))
        return dt

    def set(self, args, device):
        self.device = device
        self.anchors_per_grid = len(self.config.model.anchor_ratios) * len(self.config.model.anchor_scales)
        self.assignment = AnchorAssign(self.config, device)
        self.loss = GeneralLoss(self.config, device)
        if self.config.model.use_anchor:
            self.anchs = torch.from_numpy(generateAnchors(self.config, singleBatch=True)).float().to(device)
            self.output_number = self.anchs.shape[0]
        else:
            raise NotImplementedError('Yolov3 do not support anchor free')
        assert self.config.data.ignored_input is True, "Please set the config.data.ignored_input as True"

        self.loss_order = []
        for type in self.config.loss.reg_type:
            if 'l1' in type: self.loss_order.append('l1')
            if 'iou' in type: self.loss_order.append('iou')

    def training_loss(self,sample):
        dt = self.core(sample['imgs'])
        anchors = torch.tile(self.anchs.t(), (dt.shape[0], 1, 1))
        cls_dt = dt[:, 4:, :]
        #cls_dt = self.sigmoid(dt[:, 4:, :].clamp(-9.9,9.9))
        if 'iou' in self.loss_order:
            dt_for_iou = dt[:, :4].clone()
            dt_for_iou[:, 2:] = anchors[:, 2:] * torch.exp(dt[:, 2:4].clamp(max=50))
            dt_for_iou[:, :2] = anchors[:, :2] + dt[:, :2] * anchors[:, 2:]
        assign_result, gt = self.assignment.assign(sample['annss'])

        fin_loss = 0
        fin_loss_dict = {}
        num_pos_samples = 0
        for ib in range(len(gt)):
            dt_list = []
            gt_list = []
            assign_result_ib, gt_ib = assign_result[ib], gt[ib]
            pos_mask = torch.gt(assign_result_ib, 0.5)
            pos_neg_mask = torch.gt(assign_result_ib, -0.5)
            num_pos_samples += pos_mask.sum()
            label_pos_generate = (assign_result_ib[pos_mask] - 1).long()
            for loss_type in self.loss_order:
                if loss_type == 'iou':
                    iou_dt_ib = dt_for_iou[ib, :, pos_mask]
                    iou_gt_ib = gt_ib[label_pos_generate, :4].t()
                    dt_list.append(iou_dt_ib)
                    gt_list.append(iou_gt_ib)
                else:
                    l1_dt_ib = dt[ib, :4, pos_mask]
                    l1_gt_ib = gt_ib[label_pos_generate, :4]
                    l1_gt_ib[:, 2:] = torch.log(l1_gt_ib[:, 2:] / self.anchs[pos_mask, 2:])
                    l1_gt_ib[:, :2] = (l1_gt_ib[:, :2] - self.anchs[pos_mask, :2]) / self.anchs[pos_mask, 2:]
                    l1_gt_ib = l1_gt_ib.t()
                    dt_list.append(l1_dt_ib)
                    gt_list.append(l1_gt_ib)
            cls_all_dt_ib = cls_dt[ib, :, pos_neg_mask]
            cls_gt_ib = torch.zeros((self.config.data.numofclasses, self.output_number),
                                      dtype=torch.float32).to(self.device)
            cls_gt_ib[gt_ib[label_pos_generate,4].long(), pos_mask] = 1
            cls_gt_ib = cls_gt_ib[:, pos_neg_mask]
            obj_gt_ib = assign_result_ib[pos_neg_mask].clamp(0, 1)
            cls_all_gt_ib = torch.cat([obj_gt_ib.unsqueeze(0), cls_gt_ib], 0)
            dt_list.append(cls_all_dt_ib)
            gt_list.append(cls_all_gt_ib)

            loss, lossdict = self.loss(dt_list, gt_list)
            fin_loss += loss
            updata_loss_dict(fin_loss_dict, lossdict)
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

    @staticmethod
    def coco_parse_result(results):
        return Result.result_parse_for_json(results)

    def _result_parse(self, triple):
        '''
        flatten the results according to the format of anchors
        '''
        out = torch.zeros((triple[0].shape[0], int(5 + self.config.data.numofclasses), 0))
        if torch.cuda.is_available():
            out = out.to(self.device)
        for fp in triple:
            fp = torch.flatten(fp, start_dim=2)
            split = torch.split(fp, int(fp.shape[1] / self.anchors_per_grid), dim=1)
            out = torch.cat([out]+list(split), dim=2)
        return out

    def _debug_to_file(self, *args,**kwargs):
        if self.debug:
            with open('debug.txt', 'a') as f:
                print(*args,**kwargs,file=f)


