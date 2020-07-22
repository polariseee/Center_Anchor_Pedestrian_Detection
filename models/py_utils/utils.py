import torch
import numpy as np
from functools import partial
from collections import OrderedDict
from external import NMS
import pdb


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def anchor_inside_flags(flat_anchors,
                        valid_flags,
                        img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def parse_losses(losses):
    log_vars = OrderedDict()
    # pdb.set_trace()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def parse_det_offset(Y, cfg, nms_algorithm, score=0.1,down=4):
    seman = Y[0][0, 0, :, :]
    height = Y[1][0, 0, :, :]
    offset_y = Y[2][0, 0, :, :]
    offset_x = Y[2][0, 1, :, :]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, cfg.dataset.size_test[1]), min(y1 + h, cfg.dataset.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = NMS(boxs, cfg.test_cfg.nms_threshold, nms_algorithm)
        boxs = boxs[keep, :]
    return boxs


def parse_det_bbox(dets, cfg, nms_algorithm, score_thr=0.05):
    multi_bboxes, scores = dets[0][0], dets[0][1]
    bboxes = multi_bboxes.view(scores.size(0), -1, 4)

    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    scores = scores[valid_mask]
    scores = scores.unsqueeze(1)
    bboxes = torch.cat([bboxes, scores], dim=1)

    bboxs = bboxes.data.cpu().numpy()

    if bboxes.numel != 0:
        keep = NMS(bboxs, cfg.test_cfg.nms_threshold, nms_algorithm)
        bboxs = bboxs[keep, :]
    return bboxs
