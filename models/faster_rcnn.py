# -*- coding:utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine.functional as F
import megengine.module as M

import layers
from layers.det import resnet


class FasterRCNN(M.Module):
    """
    Implement Faster R-CNN (https://arxiv.org/abs/1506.01497).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # ----------------------- build backbone ------------------------ #
        bottom_up = getattr(resnet, cfg.backbone)(
            norm=layers.get_norm(cfg.backbone_norm), pretrained=cfg.backbone_pretrained
        )
        del bottom_up.fc
        backbone_name = cfg.backbone_name
        self.enable_asff = cfg.enable_asff

        # ----------------------- build FPN ----------------------------- #
        # if backbone_name == "FPN":
        top_block = None if len(cfg.fpn_in_features) < 4 else layers.FPNP6()
        self.backbone = layers.FPN(
            bottom_up=bottom_up,
            in_features=cfg.fpn_in_features,
            out_channels=cfg.fpn_out_channels,
            norm=cfg.fpn_norm,
            top_block=top_block,
            strides=cfg.fpn_in_strides,
            channels=cfg.fpn_in_channels,
        )
        if self.enable_asff:
            self.asff = layers.ASFF(cfg)
        # else:
        #     self.backbone = layers.BiFPN( #  仅用3层
        #         bottom_up=bottom_up,
        #         in_features=cfg.fpn_in_features,
        #         out_channels=cfg.fpn_out_channels,
        #         norm=cfg.fpn_norm,
        #         num_repeats=cfg.bifpn_repeat,
        #         strides=cfg.fpn_in_strides,
        #         in_channels=cfg.fpn_in_channels,
        #     )
        # ----------------------- build RPN ----------------------------- #
        self.rpn = layers.RPN(cfg)

        # ----------------------- build RCNN head ----------------------- #
        if cfg.enable_cascade:
            self.rcnn = layers.CascadeRCNN(cfg)
        else:
            self.rcnn = layers.RCNN(cfg)


    def preprocess_image(self, image):

        padded_image = layers.get_padded_tensor(image, 32, 0.0)
        normed_image = (
            padded_image
            - np.array(self.cfg.img_mean, dtype=np.float32)[None, :, None, None]
        ) / np.array(self.cfg.img_std, dtype=np.float32)[None, :, None, None]
        return normed_image

    def forward(self, image, im_info, gt_boxes=None):
        image = self.preprocess_image(image)

        features = self.backbone(image)

        if self.enable_asff:
            features = self.asff(features)
        if self.training:
            return self._forward_train(features, im_info, gt_boxes)
        else:
            return self.inference(features, im_info)

    def _forward_train(self, features, im_info, gt_boxes):
        rpn_rois, rpn_losses = self.rpn(features, im_info, gt_boxes)
        rcnn_losses = self.rcnn(features, rpn_rois, im_info, gt_boxes)

        total_loss = 0
        for v in rpn_losses.values():
            total_loss += v
        for v in rcnn_losses.values():
            total_loss += v
        loss_dict = {"total_loss": total_loss}
        loss_dict.update(rpn_losses)
        loss_dict.update(rcnn_losses)
        self.cfg.losses_keys = list(loss_dict.keys())
        return loss_dict

    def inference(self, features, im_info):
        rpn_rois = self.rpn(features, im_info)
        pred_boxes, pred_score = self.rcnn(features, rpn_rois, im_info)
        pred_boxes = pred_boxes.reshape(-1, 4)

        scale_w = im_info[0, 1] / im_info[0, 3]
        scale_h = im_info[0, 0] / im_info[0, 2]
        pred_boxes = pred_boxes / F.concat([scale_w, scale_h, scale_w, scale_h], axis=0)
        clipped_boxes = layers.get_clipped_boxes(
            pred_boxes, im_info[0, 2:4]
        ).reshape(-1, 4)
        return pred_score, clipped_boxes


class FasterRCNNConfig:
    # pylint: disable=too-many-statements
    def __init__(self):
        self.backbone = "resnet50"
        self.backbone_pretrained = True
        self.backbone_norm = "FrozenBN"
        self.backbone_freeze_at = 2
        self.fpn_norm = None
        self.fpn_in_features = ["res2", "res3", "res4", "res5"]
        self.fpn_in_strides = [4, 8, 16, 32]
        self.fpn_in_channels = [256, 512, 1024, 2048]
        self.fpn_out_channels = 256
        self.backbone_name = "FPN"
        self.bifpn_repeat=1
        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="coco",
            root="train2017",
            ann_file="annotations/instances_train2017.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="coco",
            root="val2017",
            ann_file="annotations/instances_val2017.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 80
        self.img_mean = [103.530, 116.280, 123.675]  # BGR
        self.img_std = [57.375, 57.120, 58.395]

        # ----------------------- rpn cfg ------------------------- #
        self.rpn_stride = [4, 8, 16, 32, 64]
        self.rpn_in_features = ["p2", "p3", "p4", "p5", "p6"]
        self.rpn_channel = 256
        self.rpn_reg_mean = [0.0, 0.0, 0.0, 0.0]
        self.rpn_reg_std = [1.0, 1.0, 1.0, 1.0]

        self.anchor_scales = [[x] for x in [32, 64, 128, 256, 512]]
        self.anchor_ratios = [[0.5, 1, 2]]
        self.anchor_offset = 0.5

        self.match_thresholds = [0.3, 0.7]
        self.match_labels = [0, -1, 1]
        self.match_allow_low_quality = True
        self.rpn_nms_threshold = 0.7
        self.num_sample_anchors = 256
        self.positive_anchor_ratio = 0.5

        # ----------------------- rcnn cfg ------------------------- #
        self.rcnn_stride = [4, 8, 16, 32]
        self.rcnn_in_features = ["p2", "p3", "p4", "p5"]
        self.rcnn_reg_mean = [0.0, 0.0, 0.0, 0.0]
        self.rcnn_reg_std = [0.1, 0.1, 0.2, 0.2]

        self.num_fc = 2
        self.fc_dim = 1024
        self.enlarge_roi = False

        self.pooling_method = "roi_align"
        self.pooling_size = (7, 7)

        self.num_rois = 512
        self.fg_ratio = 0.5
        self.fg_threshold = 0.5
        self.bg_threshold_high = 0.5
        self.bg_threshold_low = 0.0


        # ------------------------ self_dist cfg -------------------------- #

        self.enable_self_distill = False
        self.dist_tau = 1.5
        # ------------------------ asff cfg -------------------------- #
        self.enable_asff = False

        self.asff_in_features = ["p2", "p3", "p4"]
        self.asff_mid_channels = [256, 256, 256]
        # ------------------------ cascade cfg -------------------------- #
        self.enable_cascade = False

        self.num_cascade_stages = 3
        self.cascade_head_ious = (0.5, 0.6, 0.7)
        self.box_reg_weights = ((10.0, 10.0, 5.0, 5.0),
                                (20.0, 20.0, 10.0, 10.0),
                                (30.0, 30.0, 15.0, 15.0))

        # ------------------------ loss cfg -------------------------- #
        self.rpn_smooth_l1_beta = 0  # use L1 loss
        self.rcnn_smooth_l1_beta = 0  # use L1 loss
        self.num_losses = 1+2+2*self.num_cascade_stages

        # ------------------------ training cfg ---------------------- #

        self.train_image_short_size = (640, 672, 704, 736, 768, 800)
        self.train_image_max_size = 1333
        self.train_prev_nms_top_n = 2000
        self.train_post_nms_top_n = 1000

        self.basic_lr = 0.02 / 16  # The basic learning rate for single-image
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.log_interval = 20
        self.nr_images_epoch = 80000
        self.max_epoch = 54
        self.warm_iters = 500
        self.lr_decay_rate = 0.1
        self.lr_decay_stages = [42, 50]

        self.enable_ema = False
        # ------------------------ testing cfg ----------------------- #
        self.test_image_short_size = 800
        self.test_image_max_size = 1333
        self.test_prev_nms_top_n = 1000
        self.test_post_nms_top_n = 1000
        self.test_max_boxes_per_image = 100
        self.test_vis_threshold = 0.3
        self.test_cls_threshold = 0.05
        self.test_nms = 0.5
