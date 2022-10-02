import os
import sys
import time

import cv2

path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
from odcore.engine.infer import Infer as _Infer
from config import get_default_cfg
from odcore.args import get_infer_args_parser
from odcore.utils.visualization import show_bbox, _add_bbox_img, printImg
from modelzoo.build_models import BuildModel
from odcore.data.dataset import VideoDataset
from odcore.utils.misc import progressbar


class BaseInfer():
    def __init__(self, cfg, args, device):
        self.cfg = cfg
        self.args = args
        self.device = device
        builder = BuildModel(cfg)
        model = builder.build()
        model.set(args, self.device)
        self.core_infer = _Infer(cfg, args, model, self.device)

    def __call__(self, img):
        result, img = self.core_infer(img)
        result = result[0].to_ori_label()
        return result, img

    def infer(self):
        result, img = self(self.args.source)
        for i in result[:, 4:]:
            print('score:', i[0], ' category:', int(i[1]))
        if self.args.labeled:
            score = result[:, 4:]
        else:
            score = None
        show_bbox(img, result[:, :4], type='x1y1x2y2', score=score)


class VideoInfer():
    output_size = dict(ori=None, hd=(1280, 720), fhd=(1920, 1080), qhd=(2560, 1440), uhd=(3840, 2160))

    def __init__(self, cfg, args, device):
        self.file_path = args.source
        self.output_size = self.output_size[args.size]
        self.data = VideoDataset(self.file_path)
        self.core_infer = BaseInfer(cfg, args, device)
        self.format = cv2.VideoWriter_fourcc(*'XVID')
        self.pre_setting()

    def pre_setting(self):
        self.base_dir = os.path.dirname(self.file_path)
        self.file_name = os.path.basename(self.file_path)
        self.name = os.path.splitext(self.file_name)[0] + '_detect'
        self.output_file = os.path.join(self.base_dir, self.name) + '.avi'
        self.video_writer = cv2.VideoWriter(self.output_file, self.format, self.data.fps, self.output_size, True)
        self.real_output_size = self.output_size if self.output_size else self.data.size

    def infer(self):
        print('Inferencing Video:')
        print('\t-Output Size: %d X %d' % (self.real_output_size[0], self.real_output_size[1]))
        print('\t-Output File: %s' % self.output_file)
        start_time = time.time()
        for id, frame in enumerate(self.data):
            det_result, ori_frame = self.core_infer(frame)
            output_frame = _add_bbox_img(ori_frame, det_result, type='x1y1x2y2')[:, :, ::-1]
            if self.output_size:
                output_frame = cv2.resize(output_frame, self.output_size)
            self.video_writer.write(output_frame)
            progressbar(float(id + 1) / len(self.data), barlenth=40)
        self.video_writer.release()
        print('Inferencing Success in %.2f s' % (time.time() - start_time))


# ----------------------------------------------------------------------------------------------------------------------
def main(args):
    assert os.path.exists(args.source), 'Invalid source path'
    cfg = get_default_cfg()
    cfg.merge_from_files(args.conf_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    source_type = os.path.splitext(args.source)[1]
    if source_type in ['.mp4', '.avi']:
        Infer = VideoInfer
    else:
        Infer = BaseInfer

    infer = Infer(cfg, args, 0)
    infer.infer()


if __name__ == "__main__":
    args = get_infer_args_parser().parse_args()
    main(args)
