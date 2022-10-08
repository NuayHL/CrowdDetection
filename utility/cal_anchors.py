import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
os.chdir(os.path.join(path, '..'))
import time
import torch
import numpy as np
from sklearn.cluster import KMeans
from odcore.data.dataset import CocoDataset
from odcore.utils.misc import progressbar
from config import get_default_cfg

def cal_anchors(config):
    fpnlevels = config.model.fpnlevels

    kmeans = KMeans(n_clusters=len(fpnlevels), random_state=0)

    dataset = CocoDataset(config.training.train_img_anns_path,
                          config.training.train_img_path,
                          config.data, 'val')

    collecting_tick = time.time()
    bboxes = []
    for id, sample in enumerate(dataset):
        bboxes.append(sample['anns'][:, 3:5])
        progressbar((id+1)/float(len(dataset)), barlenth=40)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = np.ascontiguousarray(bboxes)

    size_kmeans_tick = time.time()
    print("CollectComplete in %.2f s" % (size_kmeans_tick - collecting_tick))

    box_size = bboxes.max(axis=1).reshape(-1, 1)
    box_ratio = bboxes[:, 1] / bboxes[:, 0]

    kmeans.fit(box_size)
    scale_kmean_tick = time.time()

    print("SizeKMeanComplete in %.2f s" % (scale_kmean_tick - size_kmeans_tick))
    print(kmeans.cluster_centers_)

    size_group_index = list()




# Copy from https://github.com/ybcc2015/DeepLearning-Utils/blob/master/Anchor-Kmeans/kmeans.py
class AnchorKmeans(object):
    """
    K-means clustering on bounding boxes to generate anchors
    """
    def __init__(self, k, max_iter=300, random_seed=None):
        self.k = k
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.n_iter = 0
        self.anchors_ = None
        self.labels_ = None
        self.ious_ = None

    def fit(self, boxes):
        """
        Run K-means cluster on input boxes.
        :param boxes: 2-d array, shape(n, 2), form as (w, h)
        :return: None
        """
        assert self.k < len(boxes), "K must be less than the number of data."

        # If the current number of iterations is greater than 0, then reset
        if self.n_iter > 0:
            self.n_iter = 0

        np.random.seed(self.random_seed)
        n = boxes.shape[0]

        # Initialize K cluster centers (i.e., K anchors)
        self.anchors_ = boxes[np.random.choice(n, self.k, replace=True)]

        self.labels_ = np.zeros((n,))

        while True:
            self.n_iter += 1

            # If the current number of iterations is greater than max number of iterations , then break
            if self.n_iter > self.max_iter:
                break

            self.ious_ = self.iou(boxes, self.anchors_)
            distances = 1 - self.ious_
            cur_labels = np.argmin(distances, axis=1)

            # If anchors not change any more, then break
            if (cur_labels == self.labels_).all():
                break

            # Update K anchors
            for i in range(self.k):
                self.anchors_[i] = np.mean(boxes[cur_labels == i], axis=0)

            self.labels_ = cur_labels

    # @staticmethod
    # def iou(boxes, anchors):
    #     """
    #     Calculate the IOU between boxes and anchors.
    #     :param boxes: 2-d array, shape(n, 2)
    #     :param anchors: 2-d array, shape(k, 2)
    #     :return: 2-d array, shape(n, k)
    #     """
    #     # Calculate the intersection,
    #     # the new dimension are added to construct shape (n, 1) and shape (1, k),
    #     # so we can get (n, k) shape result by numpy broadcast
    #     w_min = np.minimum(boxes[:, 0, np.newaxis], anchors[np.newaxis, :, 0])
    #     h_min = np.minimum(boxes[:, 1, np.newaxis], anchors[np.newaxis, :, 1])
    #     inter = w_min * h_min
    #
    #     # Calculate the union
    #     box_area = boxes[:, 0] * boxes[:, 1]
    #     anchor_area = anchors[:, 0] * anchors[:, 1]
    #     union = box_area[:, np.newaxis] + anchor_area[np.newaxis]
    #
    #     return inter / (union - inter)

    @staticmethod
    def iou(boxes, anchors):
        boxes = boxes.unsqueeze(dim=1)
        w_min = torch.min(boxes[..., 0], anchors[..., 0])
        h_min = torch.min(boxes[..., 1], anchors[..., 1])
        inter = w_min * h_min
        boxes_area = boxes[..., 0] * boxes[..., 1]
        anchors_area = anchors[..., 0] * anchors[..., 1]
        union = anchors_area + boxes_area
        return inter / (union - inter)

    def avg_iou(self):
        """
        Calculate the average IOU with closest anchor.
        :return: None
        """
        return np.mean(self.ious_[np.arange(len(self.labels_)), self.labels_])

if __name__ == "__main__":
    cfg = get_default_cfg()
    cfg.merge_from_files('cfgs/yolox_ori')
    cal_anchors(cfg)