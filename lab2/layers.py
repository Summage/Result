import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import resnet
import transforms
import utils
from resnet import *


class ProposalCreator:
    def __init__(self, mode, nms_thresh=0.7,
                 n_train_pre_nms=1000,
                 n_train_post_nms=500,
                 n_test_pre_nms=1000,
                 n_test_post_nms=500,
                 min_size=16):
        self.mode = mode
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc: torch.Tensor, score, anchor, img_size, scale: float = 1.):
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # if len(anchor) == 0:
        if not isinstance(anchor, torch.Tensor):
            if not isinstance(anchor, np.ndarray):
                anchor = np.array(anchor)
            anchor = torch.from_numpy(anchor)
        if loc.is_cuda:
            anchor = anchor.cuda()

        # transform anchor and loc to regions
        roi = utils.loc2bbox(anchor, loc)
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

        # bbox smaller than the threshold in the original pic will be disposed
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # rank rois according to their score decently
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        keep = torchvision.ops.nms(roi, score, self.nms_thresh)
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


class RegionProposalNet(nn.Module):
    def __init__(self, mode:str, in_channels=512, mid_channels=512,
                 ratios=[0.5, 1., 2.], scale_anchor=[8, 16, 32], feat_stride=16):
        super(RegionProposalNet, self).__init__()
        self.feat_stride = feat_stride
        self.proposalLayer = ProposalCreator(mode)
        fn = lambda s, r: 16*s*np.sqrt(r)
        self.anchor_prototype = \
            np.array([[fn(scale_anchor[j], ratios[i]), fn(scale_anchor[j], 1./ratios[i])]
                      for i in range(len(ratios)) for j in range(len(scale_anchor))])
        self.anchor_prototype = np.array([[-h/2, -w/2, h/2, w/2]for [h, w] in self.anchor_prototype])
        self.n_anchor = self.anchor_prototype.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, self.anchor_prototype.shape[0] * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, self.anchor_prototype.shape[0] * 4, 1, 1, 0)

    def forward(self, x, img_size, scale=1.):
        B, _, H, W = x.shape
        x = F.relu(self.conv1(x))
        loc = self.loc(x).permute(0,2,3,1).contiguous().view(B, -1, 4)
        score = self.score(x).permute(0,2,3,1).contiguous().view(B, -1, 2)
        prob_score = F.softmax(score, dim=-1)
        pos_score = prob_score[:, :, 1].contiguous().view(B, -1)
        # pat the fig with anchor prototype at every pos
        xx, yy = np.meshgrid(np.arange(0, W*self.feat_stride, self.feat_stride),
                             np.arange(0, H*self.feat_stride, self.feat_stride))
        xy = np.stack((xx.ravel(), yy.ravel(),xx.ravel(), yy.ravel(),), axis=1)
        K = xy.shape[0]
        anchor = self.anchor_prototype.reshape((1, self.n_anchor, 4)) + xy.reshape((K, 1, 4))
        anchor = anchor.reshape((K * self.n_anchor, 4))

        rois = list()
        roi_indices = list()
        for i in range(B):
            roi = self.proposalLayer(loc[i], pos_score[i], anchor, img_size, scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi)
            roi_indices.append(batch_index)
        rois = torch.cat(rois, dim=0)
        roi_indices = torch.cat(roi_indices, dim=0)

        return loc, score, rois, roi_indices, anchor


class RoIHead(nn.Module):
    def __init__(self, n_classes, roi_size, spatial_scale, classifier, in_channels):
        super(RoIHead, self).__init__()
        self.classifier = classifier
        self.n_classes = n_classes
        self.loc_cls = nn.Linear(in_channels, n_classes*4)
        self.score = nn.Linear(in_channels, n_classes)
        self.roi = torchvision.ops.RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        B = x.shape[0]
        if x.is_cuda:
            rois.to('cuda')
            roi_indices.to('cuda')

        feature_map = torch.zeros_like(rois)
        feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        pool = self.roi(x, torch.cat([roi_indices[:, None], feature_map], dim=1))
        fc = self.classifier(pool)  # 300, 2048, 1, 1
        fc = fc.view(fc.shape[0], -1)

        loc_cls = self.loc_cls(fc)
        loc_cls = loc_cls.view(B, -1, loc_cls.shape[1])
        score = self.score(fc)
        score = score.view(B, -1, score.shape[1])
        return loc_cls, score