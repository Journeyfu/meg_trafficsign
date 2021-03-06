# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import models


class CustomerConfig(models.FCOSConfig):
    def __init__(self):
        super().__init__()

        # ------------------------ dataset cfg ---------------------- #
        self.train_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/train.json",
            remove_images_without_annotations=True,
            mosaic=True,
            rand_aug=True,
        )
        self.test_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/val.json",
            test_final_ann_file="annotations/test.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 5
        # ------------------------ training cfg ---------------------- #
        self.enable_ema = True
        self.stop_mosaic_epoch = 10
        self.basic_lr = 0.02 / 16
        self.max_epoch = 24
        self.lr_decay_stages = [16, 21]
        self.nr_images_epoch = 2226
        self.warm_iters = 100
        self.log_interval = 10

        self.stride = [4, 8, 16]
        self.in_features = ["p2", "p3", "p4"]
        self.num_anchors = 1
        self.anchor_offset = 0.5
        self.object_sizes_of_interest = [
            [-1, 32], [32, 64], [64, float("inf")]
        ]
        self.fpn_in_features = ["res2", "res3", "res4"]
        self.fpn_in_strides = [4, 8, 16]
        self.fpn_in_channels = [256, 512, 1024]
        self.fpn_out_channels = 256
        self.fpn_top_in_feature = "p4"
        self.fpn_top_in_channel = 256
Net = models.FCOS
Cfg = CustomerConfig