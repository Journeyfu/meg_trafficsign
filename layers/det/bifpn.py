
import math
from typing import List

import megengine.functional as F

import megengine.module as M
import megengine as mge
import layers
from functools import partial
class SingleModule(M.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, in_channels_list, out_channels, norm
    ):
        super(SingleModule, self).__init__()

        self.out_channels = out_channels
        self.in_channels_list = in_channels_list
        # build 3-levels bifpn
        self.nodes_input_offset = [
            [1, 2],
            [0, 3],
            [1, 3, 4],
            [2, 5]
        ]
        self.nodes_strides = [8, 4, 8, 16]
        self.all_nodes_strides = [4, 8, 16, 8, 4, 8, 16]
        norm_func = None if norm is None else norm(out_channels)
        self.resample_conv_edge = list()

        for node_in_inds in self.nodes_input_offset:
            resample_conv_edge_i = list()
            for in_ind in node_in_inds:
                if self.in_channels_list[in_ind] != out_channels:
                    resample_conv = layers.Conv2d(
                        self.in_channels_list[in_ind],
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        bias=norm is None,
                        norm=norm_func,
                    )
                else:
                    resample_conv = M.Identity()
                resample_conv_edge_i.append(resample_conv)
                self.in_channels_list.append(out_channels)
            self.resample_conv_edge.append(resample_conv_edge_i)

        self.edge_weights = list()
        for node_in_inds in self.nodes_input_offset:
            weight_i = mge.Parameter(F.ones(len(node_in_inds)))
            self.edge_weights.append(weight_i)

        self.fusion_convs = list()

        for _ in self.nodes_input_offset:
            fusion_conv = layers.SaperableConvBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                padding="SAME",
                norm=norm_func,
                activation=None
            )
            self.fusion_convs.append(fusion_conv)
        self.act = lambda x: x * F.sigmoid(x)
        self.down_sampling = M.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.up_sampling = partial(F.vision.interpolate, scale_factor=2, align_corners=False)

    def forward(self, inputs): # inputs: list
        nodes_features = inputs

        for node_ind, (node_in_inds, node_stride) in enumerate(
                zip(self.nodes_input_offset, self.nodes_strides)):
            # node_ind 是 每个新结点的编号
            weights_i = F.softmax(self.edge_weights[node_ind],axis=0)
            edge_features = list()

            for num_idx, in_ind in enumerate(node_in_inds):
                # num_idx表示该结点的每个入度边
                cur_node_stride = self.all_nodes_strides[in_ind]
                edge_feature = nodes_features[in_ind] # 每个结点入度的输入feature
                resample_conv = self.resample_conv_edge[node_ind][num_idx] # 对feature的维度转换
                edge_feature = resample_conv(edge_feature)
                if cur_node_stride == node_stride * 2: # 当前输入图像小
                    edge_feature = self.up_sampling(edge_feature) # B, C, H, W
                elif cur_node_stride * 2 == node_stride: # 当前输入图像大
                    edge_feature = self.down_sampling(edge_feature)
                edge_feature = edge_feature * ( weights_i[num_idx] / weights_i.sum() + 1e-4)
                edge_features.append(edge_feature)

            node_i_feature = sum(edge_features)
            node_i_feature = self.act(node_i_feature)

            node_i_feature = self.fusion_convs[node_ind](node_i_feature)
            nodes_features.append(node_i_feature)

        assert len(nodes_features) == 7
        return nodes_features[-3:]

class BiFPN(M.Module):
    """
    BiFPN without top_block
    """
    # pylint: disable=dangerous-default-value
    def __init__(
        self, bottom_up, in_features, out_channels, norm,
        num_repeats, strides, in_channels
    ):
        assert len(in_features) == 3
        super(BiFPN, self).__init__()
        self.in_features = in_features
        norm = layers.get_norm(norm)
        self.bottom_up = bottom_up

        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in strides
        }
        self._out_features = list(sorted(self._out_feature_strides.keys())) # p2, p4, p8
        self._out_feature_channels = {k: out_channels for k in self._out_features}

        self.repeated_bifpn = list()
        for i in range(num_repeats):
            if i == 0:
                in_channels_list = in_channels
            else:
                in_channels_list = [ out_channels for _ in range(len(self._out_features))]

            self.repeated_bifpn.append(SingleModule(
                in_channels_list, out_channels, norm
            ))

    def forward(self, x):

        bottom_up_features = self.bottom_up.extract_features(x)

        feats = [bottom_up_features[f] for f in self.in_features] # list

        for bifpn in self.repeated_bifpn:
             feats = bifpn(feats)

        return dict(zip(self._out_features, feats))

    def output_shape(self):
        return {
            name: layers.ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


