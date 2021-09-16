# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import json
from collections import defaultdict
import random
import cv2
import numpy as np

from megengine.data.dataset.vision.meta_vision import VisionDataset
import megengine as mge
logger = mge.get_logger(__name__)
logger.setLevel("INFO")

def has_valid_annotation(anno, order):
    # if it"s empty, there is no annotation
    if len(anno) == 0:
        return False
    if "boxes" in order or "boxes_category" in order:
        if "bbox" not in anno[0]:
            return False
    return True


class Traffic5(VisionDataset):
    r"""
    Traffic Detection Challenge Dataset.
    """

    supported_order = (
        "image",
        "boxes",
        "boxes_category",
        "info",
    )

    def __init__(
        self, root, ann_file, remove_images_without_annotations=False, mosaic=False, *, order=None
    ):
        super().__init__(root, order=order, supported_order=self.supported_order)

        with open(ann_file, "r") as f:
            dataset = json.load(f)

        self.enable_mosaic = mosaic
        logger.info("enable_mosaic: {}".format(mosaic) )
        self.imgs = dict()
        for img in dataset["images"]:
            self.imgs[img["id"]] = img

        self.img_to_anns = defaultdict(list)
        for ann in dataset["annotations"]:
            # for saving memory
            if (
                "boxes" not in self.order
                and "boxes_category" not in self.order
                and "bbox" in ann
            ):
                del ann["bbox"]
            if "polygons" not in self.order and "segmentation" in ann:
                del ann["segmentation"]
            self.img_to_anns[ann["image_id"]].append(ann)

        self.cats = dict()
        for cat in dataset["categories"]:
            self.cats[cat["id"]] = cat

        self.ids = list(sorted(self.imgs.keys()))

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                anno = self.img_to_anns[img_id]
                # filter crowd annotations
                anno = [obj for obj in anno if obj["iscrowd"] == 0]
                anno = [
                    obj for obj in anno if obj["bbox"][2] > 0 and obj["bbox"][3] > 0
                ]
                if has_valid_annotation(anno, order):
                    ids.append(img_id)
                    self.img_to_anns[img_id] = anno
                else:
                    del self.imgs[img_id]
                    del self.img_to_anns[img_id]
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(sorted(self.cats.keys()))
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def __getitem__(self, index):

        cur_info = self.pull_item(index)
        im_h, im_w = cur_info[-1][0], cur_info[-1][1]
        yc = int(random.uniform(0.5 * im_h, 1.5 * im_h)) # 拼接的图像中心
        xc = int(random.uniform(0.5 * im_w, 1.5 * im_w))

        if self.enable_mosaic:
            aug_index = [random.randint(0, len(self) - 1) for _ in range(3)]
            aug_info = [cur_info] + [self.pull_item(i) for i in aug_index]
            mosaic_image = np.zeros((im_h*2, im_w*2, 3), dtype=np.uint8)
            mosaic_labels = []
            mosaic_bboxs = []
            target = []
            for im_i in range(4):
                aug_info_i = aug_info[im_i]
                image_i = aug_info_i[0].copy()
                bbox_i = aug_info_i[1].copy()

                h_i, w_i = aug_info_i[-1][0], aug_info_i[-1][1]

                # suffix l means large image, while s means small image in mosaic aug.
                image_i, x_l, x_r, y_l, y_r, x_scale, y_scale = self.get_mosaic_coordinate(
                    image_i, im_i, xc, yc, w_i, h_i, im_h*2, im_w*2)

                mosaic_image[y_l: y_r, x_l:x_r] = image_i

                mosaic_bboxes_i = bbox_i.copy() # xyxy

                mosaic_bboxes_i[:, 0] = mosaic_bboxes_i[:, 0] * x_scale + x_l # x
                mosaic_bboxes_i[:, 1] = mosaic_bboxes_i[:, 1] * y_scale + y_l # y
                mosaic_bboxes_i[:, 2] = mosaic_bboxes_i[:, 2] * x_scale + x_l # x
                mosaic_bboxes_i[:, 3] = mosaic_bboxes_i[:, 3] * y_scale + y_l # y

                mosaic_labels.extend(aug_info_i[2])
                mosaic_bboxs.extend(mosaic_bboxes_i)
            # "image", "boxes", "boxes_category","info"

            mosaic_bboxs = np.vstack(mosaic_bboxs)
            mosaic_labels = np.array(mosaic_labels)
            target.append(mosaic_image)
            target.append(mosaic_bboxs)
            target.append(mosaic_labels)

            info = [im_h*2, im_w*2, cur_info[-1][2], cur_info[-1][-1]]

            target.append(info)
            cur_info = tuple(target)

        return cur_info

    def pull_item(self, index):
        img_id = self.ids[index]
        anno = self.img_to_anns[img_id]
        target = []

        for k in self.order: # "image", "boxes", "boxes_category","info"
            if k == "image":
                file_name = self.imgs[img_id]["file_name"]
                path = os.path.join(self.root, file_name)
                # print(path)
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                target.append(image)
            elif k == "boxes":
                boxes = [obj["bbox"] for obj in anno]
                boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
                # transfer boxes from xywh to xyxy
                boxes[:, 2:] += boxes[:, :2]
                target.append(boxes)
            elif k == "boxes_category":
                boxes_category = [obj["category_id"] for obj in anno]
                boxes_category = [
                    self.json_category_id_to_contiguous_id[c] for c in boxes_category
                ] # 从1开始
                boxes_category = np.array(boxes_category, dtype=np.int32)
                target.append(boxes_category)
            elif k == "info":
                info = self.imgs[img_id]
                info = [info["height"], info["width"], info["file_name"], img_id]
                target.append(info)
            else:
                raise NotImplementedError
        return tuple(target)

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.ids[index]
        img_info = self.imgs[img_id]
        return img_info

    class_names = (
        "red_tl",
        "arr_s",
        "arr_l",
        "no_driving_mark_allsort",
        "no_parking_mark",
    )

    classes_originID = {
        "red_tl": 0,
        "arr_s": 1,
        "arr_l": 2,
        "no_driving_mark_allsort": 3,
        "no_parking_mark": 4,
    }


    def get_mosaic_coordinate(self, img_i, mosaic_index, xc, yc, w_i, h_i, mosaic_h, mosaic_w):
        """
        :param mosaic_index: 索引， 对应上下左右四个区域
        :param xc: 四个区域的分割点x
        :param yc: 四个区域的分割点y
        :param w:  当前图像的w
        :param h:  当前图像的h
        :param input_h: 整个图像的w
        :param input_w: 整个图像的h
        :return:
        """

        # index0 to top left part of image
        if mosaic_index == 0:
            target_shape = (xc, yc)
            img_i = cv2.resize(img_i, target_shape, interpolation=cv2.INTER_LINEAR)
            x_l, x_r, y_l, y_r = 0, xc, 0, yc
            x_scale = target_shape[0] / w_i
            y_scale = target_shape[1] / h_i

        # index1 to top right part of image
        elif mosaic_index == 1:
            target_shape = (mosaic_w - xc, yc)
            img_i = cv2.resize(img_i, target_shape, interpolation=cv2.INTER_LINEAR)
            x_l, x_r, y_l, y_r = xc, mosaic_w, 0, yc
            x_scale = target_shape[0] / w_i
            y_scale = target_shape[1] / h_i
        # index2 to bottom left part of image
        elif mosaic_index == 2:
            target_shape = (xc, mosaic_h - yc)
            img_i = cv2.resize(img_i, target_shape, interpolation=cv2.INTER_LINEAR)
            x_l, x_r, y_l, y_r = 0, xc, yc, mosaic_h
            x_scale = target_shape[0] / w_i
            y_scale = target_shape[1] / h_i
        # index2 to bottom right part of image
        elif mosaic_index == 3:
            target_shape = (mosaic_w - xc, mosaic_h-yc)
            img_i = cv2.resize(img_i, target_shape, interpolation=cv2.INTER_LINEAR)
            x_l, x_r, y_l, y_r = xc, mosaic_w,  yc, mosaic_h
            x_scale = target_shape[0] / w_i
            y_scale = target_shape[1] / h_i

        return img_i, x_l, x_r, y_l, y_r, x_scale, y_scale
