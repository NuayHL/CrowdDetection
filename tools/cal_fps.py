import os
path = os.getcwd()
os.chdir(os.path.join(path, '..'))
import torch
from torch.cuda import amp
import time
from config import get_default_cfg
from modelzoo.build_models import BuildModel
from odcore.data.dataloader import build_dataloader
from odcore.data.data_augment import Normalizer
from odcore.utils.paralle import de_parallel

def measure_fps(cfg, ckpt='',device=0, rept=1, datalenth=10000):
    print("CONFIG: %s" % cfg)
    config = get_default_cfg()
    config.merge_from_files(cfg)
    builder = BuildModel(config)
    model = builder.build()
    model.set(None, device)
    normalizer = Normalizer(config.data, device)
    dataloader = build_dataloader("CrowdHuman/annotation_val_coco_style_2000.json",
                                  "CrowdHuman/Images_val",
                                  config.data,
                                  batch_size=1, rank=-1, workers=0, task='val')


    print('Model Parameters: ', end='')
    if ckpt != '':
        print(ckpt)
        print('\t-Loading:', end=' ')
        try:
            ckpt_file = torch.load(ckpt)
            try:
                model.load_state_dict(ckpt_file['model'])
            except:
                print('FAIL')
                print('\t-Parallel Model Loading:', end=' ')
                model.load_state_dict(de_parallel(ckpt_file['model']))
            print('SUCCESS')
        except:
            print("FAIL")
            raise
    else:
        print('No ckpt for eval')

    model = model.to(device)
    model.eval()
    total_fps = []

    for exp in range(rept):
        print('========== Exp%d ==========' % (exp+1))
        warmup_num = 20
        total_img = 0
        total_inferecing_time = 0.0
        pro_bar = ProgressBar(min(len(dataloader),datalenth))

        for i, sample in enumerate(dataloader):
            if i >= datalenth:
                break
            sample['imgs'] = sample['imgs'].to(device).float() / 255
            normalizer(sample)

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():
                model.core(sample['imgs'])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i < warmup_num:
                pro_bar.update()
                continue
            total_img += 1
            total_inferecing_time += elapsed
            pro_bar.update(endstr='%f' % elapsed)
        print("total-time: %f" % total_inferecing_time)
        print("total-images: %s" % total_img)
        fps = total_img/total_inferecing_time
        print("FPS: %f\n" % fps)
        total_fps.append(fps)

    print("[Aver. FPS: %f]" % (sum(total_fps)/rept))
    print("========== Complete ==========\n")


class ProgressBar:
    def __init__(self, iters, barlenth=20, endstr=''):
        self._count = 0
        self._all = 0
        self._len = barlenth
        self._end = endstr
        if isinstance(iters, int):
            self._all = iters
        elif hasattr(iters, '__len__'):
            self._all = len(iters)
        else:
            raise NotImplementedError

    def update(self, step: int = 1, endstr=''):
        self._count += step
        if self._count == self._all:
            endstr += '\n'
        percentage = float(self._count) / self._all
        print('\r[' + '>' * int(percentage * self._len) +
              '-' * (self._len - int(percentage * self._len)) + ']',
              format(percentage * 100, '.1f'), '%',
              end=' ' + self._end + ' ' + endstr)

if __name__ == "__main__":
    measure_fps('cfgs/yolox_ori', rept=3, datalenth=120)
    measure_fps('cfgs/yolox_m', rept=3, datalenth=120)
    measure_fps('cfgs/yolox_s', rept=3, datalenth=120)
    measure_fps('cfgs/yolo_v3', rept=3, datalenth=120)
    measure_fps('cfgs/yolo_v4', rept=3, datalenth=120)
    measure_fps('cfgs/yolo_v7', rept=3, datalenth=120)
