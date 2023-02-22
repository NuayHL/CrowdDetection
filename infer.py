import os
import time
import cv2
import torch
from config import get_default_cfg
from odcore.args import get_infer_args_parser
from odcore.engine.infer import Infer as _Infer
from odcore.utils.visualization import show_bbox, _add_bbox_img, printImg
from odcore.data.dataset import VideoReader, FileImgReader
from odcore.utils.misc import progressbar
from modelzoo.build_models import BuildModel


class BaseInfer():
    def __init__(self, cfg, args, device):
        self.cfg = cfg
        self.args = args
        self.device = device
        builder = BuildModel(cfg)
        model = builder.build()
        model.set(args, self.device)
        self.core_infer = _Infer(cfg, args, model, self.device)

    def __call__(self, *img):
        results, imgs = self.core_infer(*img)
        results = [result.to_ori_label() for result in results]
        return results, imgs

    def infer(self):
        results, imgs = self(self.args.img)
        for result, img in zip(results, imgs):
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
        self.file_path = args.video
        self.resize_flag = self.output_size[args.size]
        source_type = os.path.splitext(self.file_path)[1]
        if source_type in ['.mp4', '.avi']:
            Reader = VideoReader
        else:
            Reader = FileImgReader
        self.data = Reader(self.file_path)
        self.core_infer = BaseInfer(cfg, args, device)
        self.format = cv2.VideoWriter_fourcc(*'XVID')
        self.pre_setting()

    def pre_setting(self):
        self.base_dir = os.path.dirname(self.file_path)
        self.file_name = os.path.basename(self.file_path)
        self.name = os.path.splitext(self.file_name)[0] + '_detect'
        self.output_file = os.path.join(self.base_dir, self.name) + '.avi'
        self.real_output_size = self.resize_flag if self.resize_flag else self.data.size
        self.video_writer = cv2.VideoWriter(self.output_file, self.format, self.data.fps, self.real_output_size, True)

    def infer(self, max_stacks=4):
        print('Inferencing Video:')
        print('\t-Output Size: %d X %d' % (self.real_output_size[0], self.real_output_size[1]))
        print('\t-Output File: %s' % self.output_file)
        start_time = time.time()
        max_stacks = max(max_stacks, 1)
        frame_buff = list()
        for id, frame in enumerate(self.data):
            frame_buff.append(frame)
            if len(frame_buff) >= max_stacks or id == len(self.data) - 1:
                det_results, ori_frames = self.core_infer(*frame_buff)
                frame_buff = list()
                for det_result, ori_frame in zip(det_results, ori_frames):
                    output_frame = _add_bbox_img(ori_frame, det_result, type='x1y1x2y2')[:, :, ::-1]
                    if self.resize_flag:
                        output_frame = cv2.resize(output_frame, self.real_output_size)
                    self.video_writer.write(output_frame)
            else:
                continue
            progressbar(float(id + 1) / len(self.data), barlenth=40)
        self.video_writer.release()
        print('Inferencing Success in %.2f s' % (time.time() - start_time))


# ----------------------------------------------------------------------------------------------------------------------
def main(args):
    source_type = None
    if args.img != '':
        source_type = 'img'
        source = args.img
        Infer = BaseInfer
    elif args.video != '':
        source_type = 'video'
        source = args.video
        Infer = VideoInfer
    else:
        raise NotImplementedError

    assert os.path.exists(source), 'Invalid source path'
    cfg = get_default_cfg()
    cfg.merge_from_files(args.conf_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    infer = Infer(cfg, args, 0)
    infer.infer()


if __name__ == "__main__":
    args = get_infer_args_parser().parse_args()
    main(args)
