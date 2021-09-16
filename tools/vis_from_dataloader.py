"""
modified from
https://github.com/hhaAndroid/mmdetection-mini/blob/master/tools/dataset_analyze.py
"""
import argparse
import cv2
import numpy as np
import torch
import os.path as osp
import matplotlib.pyplot as plt
import megengine.distributed as dist
from train import build_dataloader, make_parser
from tools.utils import import_from_file
from sklearn.cluster import KMeans
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')

    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-n", "--devices", default=1, type=int, help="total number of gpus for training",
    )
    parser.add_argument(
        "-b", "--batch_size", default=2, type=int, help="batch size for training",
    )
    parser.add_argument(
        "-d", "--dataset_dir", default="/data/datasets", type=str,
    )

    args = parser.parse_args()
    return args


def visualization(cfg, args):
    # stop_count 防止数据太大，要很久才能跑完
    tot_step = cfg.nr_images_epoch // (args.batch_size * dist.get_world_size())
    dataloader = iter(build_dataloader(args.batch_size, args.dataset_dir, cfg))
    print('----开始遍历数据集----')

    color = [[120, 166, 157],  [110, 76, 0],
                      [174, 57, 255], [199, 100, 0], [72, 0, 118]]

    for _ in tqdm(range(tot_step+1)):
        next_data = next(dataloader)

        image = next_data["data"] # 2, 3, h, w
        im_info = next_data["im_info"]
        gt_boxes_with_cls = next_data["gt_boxes"]
        file_names = next_data["file_names"]

        for im_i, (image, gt_boxes_with_cls, name) in enumerate(
                zip(image, gt_boxes_with_cls, file_names)):
            image = np.ascontiguousarray(image.transpose(1,2,0)).astype(np.uint8)
            valid_mask = gt_boxes_with_cls.sum(axis=-1) != 0
            num_inst = valid_mask.sum()
            gt_boxes_with_cls_valid = gt_boxes_with_cls[valid_mask]
            gt_boxes = gt_boxes_with_cls_valid[:, :4]
            gt_cls = gt_boxes_with_cls_valid[:, 4]

            for box_i in range(num_inst):
                x1,y1,x2,y2 = gt_boxes[box_i].astype(np.int32)
                cls_i = gt_cls[box_i].astype(np.int32)-1
                cv2.rectangle(image, (x1,y1), (x2,y2), color[cls_i])

            cv2.imwrite("/data/Datasets/dataset-2805/vis/aug_vis/{}".format(name), image)


if __name__ == '__main__':
    # 跑代码前，调成self.train_image_short_size=800
    args = parse_args()
    cfg = import_from_file(args.file).Cfg()
    visualization(cfg, args)

    # python3  tools/vis_from_dataloader.py -n 1
    # -f configs/fcos_res50_800size_trafficdet_demo.py -d dataset-2805