import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import os
from matplotlib import pyplot as plt
import scipy.signal


def compute_iou(bbox1, bbox2):

    # TODO Compute IoU of 2 bboxes.

    if bbox1.shape[1] != 4 or bbox2.shape[1] != 4:
        print(bbox1, bbox2)
        raise IndexError
    tl = np.maximum(bbox1[:, None, :2], bbox2[:, :2])
    br = np.minimum(bbox1[:, None, 2:], bbox2[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox1[:, 2:] - bbox1[:, :2], axis=1)
    area_b = np.prod(bbox2[:, 2:] - bbox2[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

    # End of todo

def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


def loc2bbox(anchor, loc):
    w_anc = torch.unsqueeze(anchor[:, 2] - anchor[:, 0], -1)
    h_anc = torch.unsqueeze(anchor[:, 3] - anchor[:, 1], -1)
    x_anc_ctr = torch.unsqueeze(anchor[:, 0], -1) + w_anc / 2
    y_anc_ctr = torch.unsqueeze(anchor[:, 1], -1) + h_anc / 2

    dw, dh, dx, dy = loc[:, 2::4], loc[:, 3::4], loc[:, 0::4], loc[:, 1::4]

    x_ctr, y_ctr = dx * w_anc + x_anc_ctr, dy * h_anc + y_anc_ctr
    w, h = torch.exp(dw) * w_anc, torch.exp(dh) * h_anc
    bbox_original = torch.zeros_like(loc)
    bbox_original[:, 0::4] = x_ctr - w / 2
    bbox_original[:, 1::4] = y_ctr - h / 2
    bbox_original[:, 2::4] = x_ctr + w / 2
    bbox_original[:, 3::4] = y_ctr + h / 2
    return bbox_original


class DecodeBox():
    def __init__(self, std, num_classes):
        self.std = std
        self.num_classes = num_classes+1

    def forward(self, roi_cls_locs, roi_scores, rois, h, w, nms_iou, threshold):
        roi_cls_locs = roi_cls_locs * self.std
        roi_cls_locs = roi_cls_locs.view([-1, self.num_classes, 4])
        roi = rois.view((-1,1,4)).expand_as(roi_cls_locs)
        cls_bbox = loc2bbox(roi.reshape((-1, 4)), roi_cls_locs.reshape((-1, 4)))
        cls_bbox = cls_bbox.view([-1, self.num_classes, 4])

        cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]).clamp(min=0, max=w)
        cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]).clamp(min=0, max=h)

        prob = F.softmax(roi_scores, dim=-1)

        class_conf, class_pred = torch.max(prob, dim=-1)
        conf_mask = (class_conf >= threshold)
        cls_bbox, class_conf, class_pred = \
            cls_bbox[conf_mask], class_conf[conf_mask], class_pred[conf_mask]
        output = []
        for i in range(1, self.num_classes):
            class_mask = (class_pred == i)
            cls_bbox_i = cls_bbox[class_mask, i, :]
            class_conf_i = class_conf[class_mask]
            if len(class_conf_i) == 0:
                continue
            detections_class = torch.cat([cls_bbox_i,
                                          torch.unsqueeze(class_pred[class_mask]-1, -1).float(),
                                          torch.unsqueeze(class_conf_i, -1)], -1)
            keep = torchvision.ops.nms(detections_class[:, :4], detections_class[:, -1], nms_iou)
            output.extend(detections_class[keep].cpu().numpy())
        return output


class AnchorTargetCreator:
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        # [anchor, bbox]
        ious = compute_iou(anchor, bbox)

        if len(bbox) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        # get the best match for each anchor
        argmax_ious = ious.argmax(axis=1)
        # get corresponding iou
        # max_ious = np.max(ious, axis=1)
        max_ious = np.array([ious[i][j] for i,j in zip(range(ious.shape[0]), argmax_ious)])
        # get the best match for each bbox
        gt_argmax_ious = ious.argmax(axis=0)
        # for i in range(len(gt_argmax_ious)):
        #     argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious

    def _create_label(self, anchor, bbox):
        # 1 for positive, 0 for negative, -1 should be neglected
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)

        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious) > 0:
            label[gt_argmax_ious] = 1
        # limit quantity
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        # balance pos and neg
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label


class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        iou = compute_iou(roi, bbox)

        if len(bbox) == 0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            # the best match for prop
            gt_assignment = iou.argmax(axis=1)
            # the corresponding iou
            max_iou = np.array([iou[i][j] for i,j in zip(range(iou.shape[0]), gt_assignment)])
            gt_roi_label = label[gt_assignment] + 1

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        #   sample_roi      [n_sample, ]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label


