#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import megengine.functional as F
import megengine.module as M

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv, UpSample


class YOLOPAFPN(M.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self, depth=1.0, width=1.0, in_features=("dark2", "dark3", "dark4"),
        in_channels=[64, 128, 256], depthwise=False, act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = UpSample(scale_factor=2, mode="bilinear")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            1,
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), 128, 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            64+128,
            128,
            # int(in_channels[0] * width),
            1,
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            128, 128, 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            128+128,
            128,
            # int(in_channels[1] * width),
            1,
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            128, 128, 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            128+128,
            128,
            # int(in_channels[2] * width),
            1,
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.
        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features # 128, 256, 512

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = F.concat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)
        ####################################

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = F.concat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = F.concat([p_out1, fpn_out1], 1)

        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = F.concat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)

        outputs = {"p2": pan_out2, "p3": pan_out1,  "p4": pan_out0}

        return outputs
