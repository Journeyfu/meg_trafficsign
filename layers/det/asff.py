
import megengine.functional as F
import megengine.module as M
import layers



class ASFF(M.Module):
    def __init__(self, cfg):
        super(ASFF, self).__init__()
        self.cfg = cfg

        self.in_features = cfg.asff_in_features

        self.asff_layer = dict()
        for idx, lvl in enumerate(self.in_features):
            self.asff_layer[lvl] = ASFF_LAYER(cfg, lvl)

    def forward(self, x):
        for lvl in self.in_features:
            x[lvl] = self.asff_layer[lvl](x)
        return x

class ASFF_LAYER(M.Module):
    def __init__(self, cfg, lvl):
        super(ASFF_LAYER, self).__init__()
        self.asff_mid_channels = cfg.asff_mid_channels
        self.lvl = lvl

        self.rpn_out_channel = cfg.rpn_channel
        self.in_features = cfg.asff_in_features

        lvl_ind = self.in_features.index(lvl)

        inter_channel = self.asff_mid_channels[lvl_ind]
        if self.lvl == "p4":

            self.stride_level_1 = layers.add_conv(self.rpn_out_channel, inter_channel, 3, 2)
            self.stride_level_2 = layers.add_conv(self.rpn_out_channel, inter_channel, 3, 2)
            self.expand = layers.add_conv(inter_channel, self.rpn_out_channel, 3, 1)
        elif self.lvl == "p3":
            self.compress_level_0 = layers.add_conv(self.rpn_out_channel, inter_channel, 1, 1)
            self.stride_level_2 = layers.add_conv(self.rpn_out_channel, inter_channel, 3, 2)

            self.expand = layers.add_conv(inter_channel, self.rpn_out_channel, 3, 1)
        elif self.lvl == "p2":
            self.compress_level_0 = layers.add_conv(self.rpn_out_channel, inter_channel, 1, 1)
            self.expand = layers.add_conv(inter_channel, self.rpn_out_channel, 3, 1)
        else:
            raise Exception
        compress_channel = 16
        self.weight_level_0 = layers.add_conv(inter_channel, compress_channel, 1, 1)
        self.weight_level_1 = layers.add_conv(inter_channel, compress_channel, 1, 1)
        self.weight_level_2 = layers.add_conv(inter_channel, compress_channel, 1, 1)
        self.weight_levels = M.Conv2d(compress_channel * 3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):

        if self.lvl == "p4":

            level_0_resized = inputs["p4"]
            level_1_resized = self.stride_level_1(inputs["p3"])
            level_2_downsampled_inter = F.max_pool2d(inputs["p2"], 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.lvl == "p3":
            level_0_compressed = self.compress_level_0(inputs["p4"])
            level_0_resized = F.vision.interpolate(level_0_compressed, scale_factor=2, align_corners=False)
            level_1_resized = inputs["p3"]
            level_2_resized = self.stride_level_2(inputs["p2"])
        elif self.lvl == "p2":
            level_0_compressed = self.compress_level_0(inputs["p4"])
            level_0_resized = F.vision.interpolate(level_0_compressed, scale_factor=4, align_corners=False)
            level_1_resized = F.vision.interpolate(inputs["p3"], scale_factor=2, align_corners=False)
            level_2_resized = inputs["p2"]
        else:
            raise Exception

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = F.concat((level_0_weight_v, level_1_weight_v, level_2_weight_v), axis=1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, axis=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        return self.expand(fused_out_reduced)

