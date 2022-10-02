import os
import sys

from odcore.data.dataset import VideoDataset

path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
import cv2
from infer import Infer
from config import get_default_cfg
from odcore.args import get_infer_args_parser
from odcore.utils.misc import progressbar
import time


class VideoInfer():
    output_size = dict(ori=None, hd=(1280, 720), fhd=(1920, 1080), qhd=(2560, 1440), uhd=(3840, 2160))

    def __init__(self, cfg, args, device):
        self.file_path = args.source
        self.output_size = self.output_size[args.size]
        self.data = VideoDataset(self.file_path)
        self.core_infer = Infer(cfg, args, device)
        self.format = cv2.VideoWriter_fourcc(*'XVID')

    def pre_setting(self):
        self.base_dir = os.path.dirname(self.file_path)
        self.file_name = os.path.basename(self.file_path)
        self.name = os.path.splitext(self.file_name)[0] + '_detect'
        self.output_file = os.path.join(self.base_dir, self.name) + '.avi'
        self.video_writer = cv2.VideoWriter(self.output_file, self.format, self.data.fps, self.output_size, True)
        self.real_output_size = self.output_size if self.output_size else self.data.size

    def infer(self):
        print('Inferencing Video: ')
        print('\t-Output Size: %d X %d' % (self.real_output_size[0], self.real_output_size[1]))
        print('\t-Output File: %s' % self.output_file)
        start_time = time.time()
        for id, frame in enumerate(self.data):
            input_frame = self.core_infer(frame, img_only=True)[:, :, ::-1]
            if self.output_size:
                input_frame = cv2.resize(input_frame, self.output_size)
            self.video_writer.write(input_frame)
            progressbar(float(id + 1) / len(self.data), barlenth=40)
        self.video_writer.release()
        print('Inferencing Success in %.2f s' % (time.time() - start_time))


def main(args):
    assert os.path.exists(args.source), 'Invalid source path'
    cfg = get_default_cfg()
    cfg.merge_from_files(args.conf_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    infer = VideoInfer(cfg, args, 0)
    infer.infer()


if __name__ == "__main__":
    args = get_infer_args_parser().parse_args()
    main(args)
