# -*- coding:utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import megengine.functional as F
import megengine.module as M
import numpy as np
import layers
import megengine as mge
class CascadeRCNN(M.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # roi head
        self.in_features = cfg.rcnn_in_features
        self.stride = cfg.rcnn_stride
        self.pooling_method = cfg.pooling_method
        self.pooling_size = cfg.pooling_size
        self.rpn_in_channels = cfg.fpn_out_channels
        self.num_fc = cfg.num_fc
        self.fc_dim = cfg.fc_dim
        self.cascade_head_ious = cfg.cascade_head_ious
        self.box_reg_weights = cfg.box_reg_weights
        self.dist_tau = cfg.dist_tau
        self.enable_self_distill = cfg.enable_self_distill
        # self.box_coder = layers.BoxCoder(cfg.rcnn_reg_mean, cfg.rcnn_reg_std)

        # cascade parameter

        self.enlarge_roi = cfg.enlarge_roi
        self.num_cascade_stages = cfg.num_cascade_stages
        self.box_predictor = list()
        self.box2box_transform = []
        self.proposal_matchers = []

        self.box_head = list()
        self.scale_grad = layers.ScaleGradient()
        for k in range(self.num_cascade_stages):
            box_head_i = FastRCNNFCHEAD(cfg)
            box_pred_i = FastRCNNOutputLayers(cfg)
            self.box_head.append(box_head_i)
            self.box_predictor.append(box_pred_i)
            self.box2box_transform.append(
                layers.BoxCoder(cfg.rcnn_reg_mean, cfg.rcnn_reg_std, self.box_reg_weights[k]))
            if k == 0:
                self.proposal_matchers.append(None)
            else:
                self.proposal_matchers.append(layers.Matcher([
                    self.cascade_head_ious[k]], [0, 1], allow_low_quality_matches=False)
                )

    def match_and_label_boxes(self, boxes, stage_idx, im_info, gt_targets):
        # boxes: L, 1+4
        gt_classes_list = []
        gt_boxes_list = []

        for bs_i, (info_per_image, targets_per_image) in enumerate(zip(im_info, gt_targets)):

            num_valid_instance = int(info_per_image[4])
            valid_targets_per_image = targets_per_image[:num_valid_instance, :4]
            valid_labels_per_image = targets_per_image[:num_valid_instance, 4] # 0, 1-5

            proposals_per_image = boxes[boxes[:, 0] == bs_i][:, 1:]

            match_quality_matrix = layers.get_iou(valid_targets_per_image, proposals_per_image)

            # proposal_labels are 0 or 1 表示没有/有匹配上
            matched_idxs, proposal_labels = self.proposal_matchers[stage_idx](match_quality_matrix)

            if len(targets_per_image) > 0:
                gt_classes = valid_labels_per_image[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=0)
                gt_classes[proposal_labels == 0] = 0
                gt_boxes = valid_targets_per_image[matched_idxs]
            else:
                gt_classes = F.zeros_like(matched_idxs)
                gt_boxes = F.zeros((len(proposals_per_image), 4), dtype=gt_classes.dtype)

            gt_classes_list.append(gt_classes)
            gt_boxes_list.append(gt_boxes)

        return F.concat(gt_classes_list, axis=0).detach(), F.concat(gt_boxes_list, axis=0).detach()




    def get_ground_truth(self, rpn_rois, im_info, gt_boxes):
        if not self.training:
            return rpn_rois, None, None

        return_rois = []
        return_labels = []
        return_bbox_targets = []

        # get per image proposals and gt_boxes
        for bid in range(gt_boxes.shape[0]):
            num_valid_boxes = im_info[bid, 4].astype("int32")
            gt_boxes_per_img = gt_boxes[bid, :num_valid_boxes, :]
            batch_inds = F.full((gt_boxes_per_img.shape[0], 1), bid)
            gt_rois = F.concat([batch_inds, gt_boxes_per_img[:, :4]], axis=1)
            batch_roi_mask = rpn_rois[:, 0] == bid
            # all_rois : [batch_id, x1, y1, x2, y2]
            all_rois = F.concat([rpn_rois[batch_roi_mask], gt_rois])

            overlaps = layers.get_iou(all_rois[:, 1:], gt_boxes_per_img)

            max_overlaps = overlaps.max(axis=1)
            gt_assignment = F.argmax(overlaps, axis=1).astype("int32")
            labels = gt_boxes_per_img[gt_assignment, 4]

            # ---------------- get the fg/bg labels for each roi ---------------#
            fg_mask = (max_overlaps >= self.cfg.fg_threshold) & (labels >= 0)
            bg_mask = (
                (max_overlaps >= self.cfg.bg_threshold_low)
                & (max_overlaps < self.cfg.bg_threshold_high)
            )

            num_fg_rois = int(self.cfg.num_rois * self.cfg.fg_ratio)
            fg_inds_mask = layers.sample_labels(fg_mask, num_fg_rois, True, False)
            num_bg_rois = int(self.cfg.num_rois - fg_inds_mask.sum())
            bg_inds_mask = layers.sample_labels(bg_mask, num_bg_rois, True, False)

            labels[bg_inds_mask] = 0 # 背景是0

            keep_mask = fg_inds_mask | bg_inds_mask
            labels = labels[keep_mask].astype("int32")
            rois = all_rois[keep_mask]
            bbox_targets = gt_boxes_per_img[gt_assignment[keep_mask], :4]
            # bbox_targets = self.box2box_transform[0].encode(rois[:, 1:], target_boxes)
            # bbox_targets = bbox_targets.reshape(-1, 4)

            return_rois.append(rois)
            return_labels.append(labels)
            return_bbox_targets.append(bbox_targets)

        return (
            F.concat(return_rois, axis=0).detach(),
            F.concat(return_labels, axis=0).detach(),
            F.concat(return_bbox_targets, axis=0).detach(),
        )

    def forward(self, fpn_fms, rcnn_rois, im_info=None, gt_targets=None):
        # gt_targets: Bs,MAX_BOX,5(4+1)
        rcnn_rois, labels, bbox_targets = self.get_ground_truth(
            rcnn_rois, im_info, gt_targets
        )
        # rcnn_rois: batch_ind, box_info(4)
        fpn_fms = [fpn_fms[x] for x in self.in_features]
        if self.training:
            losses = self.cascade_forward(fpn_fms, rcnn_rois, im_info,
                                          (labels, bbox_targets), gt_targets)
            return losses
        else:
            pred_bbox, pred_scores = self.cascade_forward(fpn_fms, rcnn_rois, im_info)
            return pred_bbox, pred_scores

    def cascade_forward(self, fpn_fms, rcnn_rois, im_info, proposals_info=None, gt_targets=None):
        head_outputs = []
        image_sizes = [(int(x[0]), int(x[1])) for x in im_info] # resize后没有padded的box大小, (H, W)
        valid_index = [[],[]]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # 原始图像尺寸
                rcnn_rois, non_empty_masks = self.create_proposals_from_boxes(
                    head_outputs[-1].predict_boxes_with_split(), image_sizes
                )
                if self.training:
                    for i in range(k):
                        if len(valid_index[i]) == 0:
                            valid_index[i] = F.arange(len(rcnn_rois), device=rcnn_rois.device, dtype="int32")
                        valid_index[i] = valid_index[i][non_empty_masks]
                    proposals_info = self.match_and_label_boxes(rcnn_rois, k, im_info, gt_targets)

            head_outputs.append(self.run_stage(fpn_fms, rcnn_rois, k, image_sizes, proposals_info))

        if self.training:
            losses = {}
            for stage, output in enumerate(head_outputs):
                stage_losses = output.losses()
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            if self.enable_self_distill:
                self_dist_loss = self.soft_label_distillation(head_outputs, valid_index)
                losses.update(self_dist_loss)
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h.predict_probs() for h in head_outputs]
            # Average the scores across heads
            pred_scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ][0] # 因为bs=1
            # Use the boxes of the last head
            pred_bbox = head_outputs[-1].predict_boxes()

            return pred_bbox, pred_scores

    def soft_label_distillation(self, head_outputs, valid_index):

        final_pred = F.softmax(head_outputs[-1].pred_class_logits/ self.dist_tau, axis=1)
        loss = {}
        for stage_i, head_outputs_i in enumerate(head_outputs[:-1]):

            pre_pred_logsoftmax = F.logsoftmax(head_outputs_i.pred_class_logits/self.dist_tau, axis=1)
            assert len(valid_index[stage_i]) == len(final_pred)
            loss["self_dist_stage{}".format(stage_i+1)] = 10 * layers.kl_div(pre_pred_logsoftmax[valid_index[stage_i]], final_pred)
 
        return loss

    def create_proposals_from_boxes(self, boxes, image_sizes):

        proposal_boxes = []
        non_empty_masks = []
        for bs_i, (boxes_per_image, image_size) in enumerate(zip(boxes, image_sizes)):
            boxes_per_image = layers.get_clipped_boxes(
                boxes_per_image.detach(), image_size
            )
            bs_ch = F.full((len(boxes_per_image), 1),value=bs_i,
                           dtype="int32", device=boxes_per_image.device)
            boxes_per_image = F.concat([bs_ch, boxes_per_image], axis=1)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                non_empty_mask = (boxes_per_image[:, 1:] != 0).sum(axis=1) > 0
                boxes_per_image = boxes_per_image[non_empty_mask]
                non_empty_masks.append(non_empty_mask)

            proposal_boxes.append(boxes_per_image)
        proposal_boxes = F.concat(proposal_boxes, axis=0)
        if self.training:
            non_empty_masks = F.concat(non_empty_masks, axis=0)
        return proposal_boxes, non_empty_masks

    def run_stage(self, fpn_fms, rcnn_rois, stage_idx, image_sizes, proposals_info=None):

        if self.enlarge_roi:

            batch_ind, batch_rois = rcnn_rois[:, 0:1], rcnn_rois[:, 1:]
            batch_rois_mean = (batch_rois[:, :2] + batch_rois[:, 2:])/2
            batch_rois_mean = F.concat([batch_rois_mean, batch_rois_mean], axis=1)
            # 1.5 为扩大倍数
            batch_rois_trans = (batch_rois - batch_rois_mean)*1.5 + batch_rois_mean
            boxes_batch = []
            for i in range(int(F.max(batch_ind))+1):
                boxes_per_image = layers.get_clipped_boxes(
                    batch_rois_trans[batch_ind[:, 0] == i], image_sizes[i]
                )
                boxes_batch.append(boxes_per_image)

            boxes_batch = F.concat(boxes_batch, axis=0)
            rcnn_rois = F.concat([batch_ind, boxes_batch], axis=1)

        pool_features = layers.roi_pool(
            fpn_fms, rcnn_rois, self.stride, self.pooling_size, self.pooling_method,
        ) # (1024, 256, 7, 7)
        pool_features = self.scale_grad(pool_features, mge.tensor(1.0 / self.num_cascade_stages))
        box_features = self.box_head[stage_idx](pool_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor[stage_idx](box_features)
        # pred_class_logits: 1024, 6
        # pred_proposal_deltas: 1024, cfg.num_classes * 4
        del box_features
        # 根据rcnn_rois 确定每个batch的数量
        outputs = FastRCNNOutputs(
            self.cfg,
            self.box2box_transform[stage_idx],
            pred_class_logits,
            pred_proposal_deltas,
            rcnn_rois,
            proposals_info=proposals_info
        )
        return outputs

class FastRCNNOutputs():
    def __init__(self, cfg, bbox_transform, pred_class_logits, pred_proposal_deltas, rcnn_rois,
                 proposals_info=None):
        self.cfg = cfg
        self.rcnn_rois = rcnn_rois
        self.pred_class_logits = pred_class_logits # 1024, (class+1)
        self.pred_proposal_deltas = pred_proposal_deltas # 1024, 4
        self.bbox_transform = bbox_transform
        bs = int(rcnn_rois[:, 0].max())

        self.nums_instance_per_image = [ (rcnn_rois[:, 0]==i).sum() for i in range(bs)]
        if proposals_info is not None:
            self.gt_labels = proposals_info[0]
            self.gt_targets = proposals_info[1]
    def losses(self):
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def smooth_l1_loss(self):
        # loss for rcnn regression

        # deltas可能为0, 因为roi插入了gt
        bbox_deltas = self.bbox_transform.encode(self.rcnn_rois[:, 1:], self.gt_targets).reshape(-1, 4)

        pred_offsets = self.pred_proposal_deltas.reshape(-1, 4)

        num_samples = self.gt_labels.shape[0]
        fg_mask = self.gt_labels > 0
        if fg_mask.sum() == 0:
            loss_rcnn_bbox = pred_offsets.sum() * 0.
        else:
            loss_rcnn_bbox = layers.smooth_l1_loss(
                pred_offsets[fg_mask],
                bbox_deltas[fg_mask],
                self.cfg.rcnn_smooth_l1_beta,
            ).sum() / F.maximum(num_samples, 1)
        return loss_rcnn_bbox

    def softmax_cross_entropy_loss(self):
        return F.loss.cross_entropy(self.pred_class_logits, self.gt_labels, axis=1)

    def predict_boxes(self):

        pred_offsets = self.pred_proposal_deltas.reshape(-1, 4)
        return self.bbox_transform.decode(self.rcnn_rois[:, 1:5], pred_offsets)


    def predict_probs(self):
        pred_scores = F.softmax(self.pred_class_logits, axis=1)[:, 1:]
        return F.split(pred_scores, self.nums_instance_per_image, axis=0)

    def predict_boxes_with_split(self):
        # megengine的split传入的list的个数=分段数-1
        pred_box_delta = self.predict_boxes().detach()
        return F.split(pred_box_delta, self.nums_instance_per_image, axis=0)

class FastRCNNFCHEAD(M.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rpn_in_channels = cfg.fpn_out_channels
        self.pooling_size = cfg.pooling_size
        self.num_fc = cfg.num_fc
        self.fc_dim = cfg.fc_dim

        in_ch = self.rpn_in_channels * self.pooling_size[0] * self.pooling_size[1]
        self.fc_list = list()

        for i in range(self.num_fc):
            self.fc_list.append(M.Linear(in_ch, self.fc_dim))
            in_ch = self.fc_dim

        for l in self.fc_list:
            M.init.normal_(l.weight, std=0.01)
            M.init.fill_(l.bias, 0)

    def forward(self, x):
        if len(self.fc_list):
            for i, layer in enumerate(self.fc_list):
                if i == 0:
                    x = F.flatten(x, start_axis=1)
                x = F.relu(layer(x))
        return x

class FastRCNNOutputLayers(M.Module):
    def __init__(self, cfg):
        super().__init__()
        # box predictor
        self.pred_cls = M.Linear(cfg.fc_dim, cfg.num_classes+1)
        self.pred_delta = M.Linear(cfg.fc_dim, 4)
        M.init.normal_(self.pred_cls.weight, std=0.01)
        M.init.normal_(self.pred_delta.weight, std=0.001)
        for l in [self.pred_cls, self.pred_delta]:
            M.init.fill_(l.bias, 0)

    def forward(self, x):
        pred_logits = self.pred_cls(x)
        pred_offsets = self.pred_delta(x)
        return pred_logits, pred_offsets