# -*- coding: utf-8 -*-
# Copyright 2019 - present, Facebook, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2021 Megvii Inc. All rights reserved.
# ---------------------------------------------------------------------
from collections import namedtuple

import megengine.module as M
import megengine.functional as F

import math


class Conv2d(M.Conv2d):
    """
    A wrapper around :class:`megengine.module.Conv2d`.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to
        `megengine.module.Conv2d`.

        Args:
            norm (M.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation
                function
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Conv2dSamePadding(M.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)

        self.padding_mode = kwargs.pop("padding", None)

        if self.padding_mode is None:
            self.padding_mode = 0

        if isinstance(self.padding_mode, str):
            assert self.padding_mode == "SAME"
            super().__init__(*args, **kwargs, padding=0)
        else:
            super().__init__(*args, **kwargs, padding=self.padding_mode)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if isinstance(self.padding_mode, str):

            # TODO： 确定shape维度
            input_h, input_w = x.shape[-2:]
            stride_h, stride_w = self.stride
            kernel_h, kernel_w = self.kernel_size
            dilation_h, dilation_w = self.dilation

            output_h = math.ceil(input_h / stride_h)
            output_w = math.ceil(input_w / stride_w)
            padding_h = max(0, (output_h-1)*stride_h + (kernel_h -1) * dilation_h + 1 - input_h)
            padding_w = max(0, (output_w-1)*stride_w + (kernel_w -1) * dilation_w + 1 - input_w)
            left = padding_w // 2
            right = padding_w - left
            top = padding_h // 2
            bottom = padding_h - top

            x = self.pad(x, [left, top, right, bottom])

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def pad(self, x, pad_list, pad_value=0):
        # pad_list: l, t, r, b
        batch, chl, t_height, t_width = x.shape
        l_pad, t_pad, r_pad, b_pad = pad_list
        padded_height = t_height + t_pad + b_pad
        padded_width = t_width + l_pad + r_pad

        padded_x = F.full(
            (batch, chl, padded_height, padded_width), pad_value, dtype=x.dtype
        )
        padded_x[:, :, t_pad:-b_pad, l_pad:-r_pad] = x

        return padded_x


class SaperableConvBlock(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, norm=None, activation=None):
        super().__init__()
        self.norm = norm
        self.activation = activation
        self.depthwise_conv = Conv2dSamePadding(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                dilation=dilation,
                                                groups=in_channels,
                                                bias=False)
        self.pointwise_conv = Conv2dSamePadding(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                dilation=1,
                                                groups=1,
                                                bias=bias)
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    Useful for getting the modules output channels when building the graph.
    """

    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)