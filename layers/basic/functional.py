# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Optional

import numpy as np
import megengine.module as M
import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
from megengine import Tensor
from megengine.autodiff import Function

def get_padded_tensor(
    array: Tensor, multiple_number: int = 32, pad_value: float = 0
) -> Tensor:
    """ pad the nd-array to multiple stride of th e

    Args:
        array (Tensor):
            the tensor with the shape of [batch, channel, height, width]
        multiple_number (int):
            make the height and width can be divided by multiple_number
        pad_value (int): the value to be padded

    Returns:
        padded_array (Tensor)
    """
    batch, chl, t_height, t_width = array.shape
    padded_height = (
        (t_height + multiple_number - 1) // multiple_number * multiple_number
    )
    padded_width = (t_width + multiple_number - 1) // multiple_number * multiple_number

    padded_array = F.full(
        (batch, chl, padded_height, padded_width), pad_value, dtype=array.dtype
    )

    ndim = array.ndim
    if ndim == 4:
        padded_array[:, :, :t_height, :t_width] = array
    elif ndim == 3:
        padded_array[:, :t_height, :t_width] = array
    else:
        raise Exception("Not supported tensor dim: %d" % ndim)
    return padded_array


def safelog(x, eps=None):
    if eps is None:
        eps = np.finfo(x.dtype).eps
    return F.log(F.maximum(x, eps))


def batched_nms(
    boxes: Tensor, scores: Tensor, idxs: Tensor, iou_thresh: float, max_output: Optional[int] = None
) -> Tensor:
    r"""
    Performs non-maximum suppression (NMS) on the boxes according to
    their intersection-over-union (IoU).

    :param boxes: tensor of shape `(N, 4)`; the boxes to perform nms on;
        each box is expected to be in `(x1, y1, x2, y2)` format.
    :param iou_thresh: ``IoU`` threshold for overlapping.
    :param idxs: tensor of shape `(N,)`, the class indexs of boxes in the batch.
    :param scores: tensor of shape `(N,)`, the score of boxes.
    :return: indices of the elements that have been kept by NMS.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor

        x = np.zeros((100,4))
        np.random.seed(42)
        x[:,:2] = np.random.rand(100,2) * 20
        x[:,2:] = np.random.rand(100,2) * 20 + 100
        scores = tensor(np.random.rand(100))
        idxs = tensor(np.random.randint(0, 10, 100))
        inp = tensor(x)
        result = batched_nms(inp, scores, idxs, iou_thresh=0.6)
        print(result.numpy())

    Outputs:

    .. testoutput::

        [75 41 99 98 69 64 11 27 35 18]

    """
    assert (
        boxes.ndim == 2 and boxes.shape[1] == 4
    ), "the expected shape of boxes is (N, 4)"
    assert scores.ndim == 1, "the expected shape of scores is (N,)"
    assert idxs.ndim == 1, "the expected shape of idxs is (N,)"
    assert (
        boxes.shape[0] == scores.shape[0] == idxs.shape[0]
    ), "number of boxes, scores and idxs are not matched"

    idxs = idxs.detach()
    max_coordinate = boxes.max()
    offsets = idxs.astype("float32") * (max_coordinate + 1)
    boxes = boxes + offsets.reshape(-1, 1)
    return F.nn.nms(boxes, scores, iou_thresh, max_output)


def all_reduce_mean(array: Tensor) -> Tensor:
    if dist.get_world_size() > 1:
        array = dist.functional.all_reduce_sum(array) / dist.get_world_size()
    return array


class ScaleGradient(Function):


    def forward(self, input, scale):
        self.scale = scale
        return input


    def backward(self, grad_output):
        return grad_output * self.scale, mge.tensor(0, device=grad_output.device)




def add_conv(in_ch, out_ch, kernel_size, stride):
    module_list = list()
    pad = (kernel_size - 1) // 2
    module_list.append(M.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=False))
    module_list.append(M.BatchNorm2d(out_ch))
    module_list.append(M.LeakyReLU(0.1))
    return M.Sequential(*module_list)

# def np_kldivloss(input, target, log_target, reduction='mean'):
#     if log_target:
#         output = np.exp(target)*(target - input)
#     else:
#         output_pos = target*(np.log(target) - input)
#         zeros = np.zeros_like(input)
#         output = np.where(target>0, output_pos, zeros)
#     if reduction == 'mean':
#         return np.mean(output)
#     elif reduction == 'sum':
#         return np.sum(output)
#     else:
#         return output

def kl_div(logsoftmax_pred, softmax_target):

    output_pos = softmax_target * (F.log(softmax_target) - logsoftmax_pred)

    return F.mean(output_pos)