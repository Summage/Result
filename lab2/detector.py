import colorsys
import copy
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
import resnet
import transforms
import utils
from layers import *
from resnet import *
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# TODO Design the detector.
# tips: Use pretrained `resnet` as backbone.


class fasterRCNN(nn.Module):
    def __init__(self, mode, extractor, classifier, rpn, head,  img_size=600,
                 loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super(fasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.mode = mode
        self.resize = transforms.Resize(img_size)
        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]
        feature = self.extractor(x)
        _, _, rois, roi_indices, _ = self.rpn(feature, img_size, scale)
        locs_cls, score = self.head(feature, rois, roi_indices, img_size)
        return locs_cls, score, rois, roi_indices

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class trainer(nn.Module):
    def __init__(self, model, optimizer):
        super(trainer, self).__init__()
        self.model = model
        self.rpn_sigma = 1
        self.roi_sigma = 1
        self.anchor_target_creator = utils.AnchorTargetCreator()
        self.proposal_target_creator = utils.ProposalTargetCreator()

        self.optimizer = optimizer

    def forward(self, imgs, bboxes, labels, scale):
        n = imgs.shape[0]
        img_size = imgs.shape[2:]
        base_feature = self.model.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.model.rpn(base_feature, img_size, scale)
        losses = 0
        correct_roi = 0
        total_roi = 0
        labels = np.array(labels)
        bboxes = np.array(bboxes)
        for i in range(n):
            bbox = bboxes[i].reshape(1, 4)
            label = np.array(labels[i]).reshape(1, )
            rpn_loc = rpn_locs[i]
            rpn_score = rpn_scores[i]
            roi = rois[roi_indices == i]
            feature = base_feature[i]

            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor, img_size)
            gt_rpn_loc = torch.Tensor(gt_rpn_loc)
            gt_rpn_label = torch.Tensor(gt_rpn_label).long()

            if rpn_loc.is_cuda:
                gt_rpn_loc = gt_rpn_loc.cuda()
                gt_rpn_label = gt_rpn_label.cuda()

            rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label)
            sample_roi = torch.Tensor(sample_roi)
            gt_roi_loc = torch.Tensor(gt_roi_loc)
            gt_roi_label = torch.Tensor(gt_roi_label).long()
            sample_roi_index = torch.zeros(len(sample_roi))

            if feature.is_cuda:
                sample_roi = sample_roi.cuda()
                sample_roi_index = sample_roi_index.cuda()
                gt_roi_loc = gt_roi_loc.cuda()
                gt_roi_label = gt_roi_label.cuda()

            roi_cls_loc, roi_score = self.model.head(torch.unsqueeze(feature, 0), sample_roi,
                                                     sample_roi_index, img_size)

            n_sample = roi_cls_loc.size()[1]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

            roi_loc_loss = _fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score[0], gt_roi_label)
            total_roi += gt_roi_label.shape[0]
            with torch.no_grad():
                tmp = roi_score.numpy()[0]
                tmp = np.argmax(tmp, axis=-1)
                correct_roi += np.sum(tmp == gt_rpn_label)
            losses += rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
        return losses/n, correct_roi, total_roi

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses, c, t = self.forward(imgs, bboxes, labels, scale)
        losses.backward()
        self.optimizer.step()
        return losses, c, t


def _smooth_l1_loss(x, t, sigma):
    sigma_squared = sigma ** 2
    regression_diff = (x - t)
    regression_diff = regression_diff.abs()
    regression_loss = torch.where(
        regression_diff < (1. / sigma_squared),
        0.5 * sigma_squared * regression_diff ** 2,
        regression_diff - 0.5 / sigma_squared
    )
    return regression_loss.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    pred_loc = pred_loc[gt_label > 0]
    gt_loc = gt_loc[gt_label > 0]

    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, sigma)
    num_pos = (gt_label > 0).sum().float()
    loc_loss /= torch.max(num_pos, torch.ones_like(num_pos))
    return loc_loss


class Detector:
    def __init__(self, classLabels, backbone='resnet50', lengths=(2048 * 4 * 4, 2048, 512)):
        self.mode = 'train'
        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False
        self.normalise = lambda x: \
            F.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_classes = len(classLabels)
        self.confidence = 0.5
        self.nm_thresh = 0.6
        self.classLabel = classLabels
        self.backbone = eval(backbone)
        self.decodeBox = utils.DecodeBox(torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes+1)[None],
                                         self.num_classes)
        net = self.backbone = self.backbone(pretrained=True, progress=True)
        extractor = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                  net.layer1, net.layer2, net.layer3)
        classifier = nn.Sequential(net.layer4, net.avgpool)
        self.model = fasterRCNN('predict', extractor, classifier,
                                RegionProposalNet(self.mode, 1024, 512),
                                RoIHead(len(self.classLabel)+1, roi_size=14, spatial_scale=1.,
                                        classifier=classifier, in_channels=lengths[-2]))
        hsv_tuples = [(x / len(self.classLabel), 1., 1.)
                      for x in range(len(self.classLabel))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect(self, img):
        ori_img = copy.deepcopy(img)
        # img = self.normalise(img)
        img = np.expand_dims(img, 0)
        with torch.no_grad():
            if self.cuda:
                img = img.cuda()
            img_size = img.shape[2:]
            feature = self.model.extractor(img)
            _, _, rois, roi_indices, _ = self.model.rpn(feature, img_size, 1.)
            roi_cls_locs, roi_scores = self.model.head(feature, rois, roi_indices, img_size)
            outputs = self.decodeBox.forward(roi_cls_locs[0], roi_scores[0], rois,
                                             self.model.resize.size, self.model.resize.size,
                                             self.nm_thresh, self.confidence)
            if len(outputs) == 0:
                return ori_img
            outputs = np.array(outputs)
            bbox = outputs[:, :4]
            label = outputs[:, 4]
            conf = outputs[:, 5]

            bbox[:, 0::2] = (bbox[:, 0::2])
            bbox[:, 1::2] = (bbox[:, 1::2])

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(img)[1] + 0.5).astype('int32'))

        thickness = 4

        image = ori_img
        for i, c in enumerate(label):
            predicted_class = self.classLabel[int(c)]
            score = conf[i]

            left, top, right, bottom = bbox[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def predict(self, img):
        img = self.normalise(img)
        img = np.transpose(img, (2, 0, 1))
        with torch.no_grad:
            if self.cuda:
                img = img.cuda()
            roi_cls_locs, roi_scores, rois, _ = self.model(img)
            outputs = self.decodeBox.forward(roi_cls_locs[0], roi_scores[0], rois,
                                             self.model.resize.size, self.model.resize.size,
                                             self.nm_thresh, self.confidence)
            outputs = np.array(outputs)
            # bbox = outputs[:, :4]
            # label = outputs[:, 4]
            # conf = outputs[:, 5]
        return outputs[:, :-1]


# End of todo

if __name__ == '__main__':
    from torch.utils import data
    from tvid import TvidDataset
    dataset = TvidDataset(root='./tiny_vid', mode='test')
    detector = Detector(classLabels=dataset.label, backbone='resnet50',
                        lengths=(2048 * 4 * 4, 2048, 512))
    modelPath = 'Total_Loss0.5841_30.pth'
    state_dict = torch.load(modelPath, map_location='cpu')
    detector.model.load_state_dict(state_dict)

    print('{} model, anchors, and classes loaded.'.format(modelPath))
    print(detector.classLabel)
    while True:
        item = input('key in the num of img that you want to detect')
        X, y = dataset[int(item)]
        img, bbox = X[0], X[1]
        detector.detect(img)
        print(X[1])
        print(y)
