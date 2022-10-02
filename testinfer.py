import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
import matplotlib
import cv2
from infer import Infer
from odcore.utils.visualization import show_bbox, _add_bbox_img, printImg
from config import get_default_cfg
from odcore.args import get_infer_args_parser
from odcore.utils.misc import progressbar

matplotlib.use('TKAgg')

class VideoDataset():
    def __init__(self, video_path):
        self.path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.lenth = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        return self

    def __next__(self):
        success, frame = self.cap.read()
        if not success:
            self.cap.release()
            raise StopIteration
        return frame[:, :, ::-1]

    def __len__(self):
        return self.lenth

class VideoInfer():
    def __init__(self, cfg, args, device):
        self.file_path = args.source
        self.data = VideoDataset(self.file_path)
        self.infer = Infer(cfg, args, device)
        self.format = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter('testing.avi', self.format, self.data.fps, self.data.size, True)

    def detect(self):
        for id, frame in enumerate(self.data):
            self.video_writer.write(self.infer(frame, img_only=True)[:, :, ::-1])
            progressbar(float(id+1)/len(self.data), barlenth=40)
        self.video_writer.release()

def main(args):
    assert os.path.exists(args.source),'Invalid source path'
    cfg = get_default_cfg()
    cfg.merge_from_files(args.conf_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    infer = VideoInfer(cfg, args, 0)
    infer.detect()

if __name__ == "__main__":
    args = get_infer_args_parser().parse_args()
    main(args)