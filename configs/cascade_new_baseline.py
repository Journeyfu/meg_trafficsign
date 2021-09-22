# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import models


class CustomerConfig(models.FasterRCNNConfig):
    def __init__(self):
        super().__init__()

        # ------------------------ dataset cfg ---------------------- #
        self.train_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/train.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/val.json",
            test_final_ann_file="annotations/test.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 5
        self.cascade_head_ious = (0.5, 0.7, 0.9)
        self.anchor_scales = [[x] for x in [32, 64, 128]]
        self.fpn_stride = [4, 8, 16]
        self.fpn_in_features = ["res2", "res3", "res4"]
        self.fpn_in_strides = [4, 8, 16]
        self.fpn_in_channels = [256, 512, 1024]

        self.rpn_stride = [4, 8, 16]
        self.rpn_in_features = ["p2", "p3", "p4"]
        self.rpn_channel = 256

        self.rcnn_stride = [4, 8, 16]
        self.rcnn_in_features = ["p2", "p3", "p4"]
        # ------------------------ training cfg ---------------------- #
        self.enable_cascade=True

        self.stop_mosaic_epoch = 18
        self.basic_lr = 0.02 / 16
        self.max_epoch = 24
        self.lr_decay_stages = [16, 21]
        self.nr_images_epoch = 2226
        self.warm_iters = 100
        self.log_interval = 10

Net = models.FasterRCNN
Cfg = CustomerConfig
