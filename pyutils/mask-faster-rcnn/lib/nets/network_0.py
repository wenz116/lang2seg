# --------------------------------------------------------
# Pytorch Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ruotian Luo & Licheng Yu
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import utils.timer

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from layer_utils.roi_pooling.roi_pool import RoIPoolFunction

from model.config import cfg

import tensorboardX as tb 

from scipy.misc import imresize

class Network(nn.Module):
  def __init__(self, batch_size=1):
    nn.Module.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1./16, ]
    self._batch_size = batch_size
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._act_summaries = {}
    self._score_summaries = {}
    self._event_summaries = {}
    self._image_gt_summaries = {}
    self._variables_to_fix = {}

  #######################
  # tensorboard functions
  #######################
  def _add_gt_image(self):
    # add back mean
    image = self._image_gt_summaries['image'] + cfg.PIXEL_MEANS
    image = imresize(image[0], self._im_info[0][:2] / self._im_info[0][2]) # assume we only have 1 image
    # BGR to RGB (opencv uses BGR)
    self._gt_image = image[np.newaxis, :,:,::-1].copy(order='C')

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    self._add_gt_image()
    image = draw_bounding_boxes(\
                      self._gt_image, self._image_gt_summaries['gt_boxes'], self._image_gt_summaries['im_info'])

    return tb.summary.image('GROUND_TRUTH', torch.from_numpy(image[0].astype('float32')/ 255.0).permute(2,0,1))

  def _add_act_summary(self, key, tensor):
    return tb.summary.histogram('ACT/' + key + '/activations', tensor.data.cpu().numpy(), bins='auto'),
    tb.summary.scalar('ACT/' + key + '/zero_fraction',
                      (tensor.data == 0).float().sum() / tensor.numel())

  def _add_score_summary(self, key, tensor):
    return tb.summary.histogram('SCORE/' + key + '/scores', tensor.data.cpu().numpy(), bins='auto')

  def _add_train_summary(self, key, var):
    return tb.summary.histogram('TRAIN/' + key, var.data.cpu().numpy(), bins='auto')

  #######################
  # model layers
  #######################
  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_top_layer(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                     self._feat_stride, self._anchors, self._num_anchors)
    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_layer(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors)

    return rois, rpn_scores

  def _roi_pool_layer(self, bottom, rois):
    return RoIPoolFunction(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1. / 16.)(bottom, rois)

  def _crop_pool_layer(self, bottom, rois, max_pool=True):
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()

    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
      pre_pool_size = cfg.POOLING_SIZE * 2
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
      crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)
      crops = F.max_pool2d(crops, 2, 2)
    else:
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
      crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)
    
    return crops

  def _crop_pool_layer_align(self, bottom, rois, im_info, max_pool=True):
    """
    Note we use original im_info and rois to compute normalized grids,
    instead of layer3's size and boxes/16. (I think my way is more accurate for RoIAlign.)
    """
    rois = rois.detach()
    x1 = rois[:, 1::4]
    y1 = rois[:, 2::4]
    x2 = rois[:, 3::4]
    y2 = rois[:, 4::4]
    height, width = float(im_info[0][0]), float(im_info[0][1])  # use original height and width

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
      pre_pool_size = cfg.POOLING_SIZE * 2
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
      crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)
      crops = F.max_pool2d(crops, 2, 2)
    else:
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
      crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)
    
    return crops

  def _anchor_target_layer(self, rpn_cls_score):
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
      anchor_target_layer(
      rpn_cls_score.data, self._gt_boxes.data.cpu().numpy(), self._im_info, self._feat_stride, self._anchors.data.cpu().numpy(), self._num_anchors)

    rpn_labels = Variable(torch.from_numpy(rpn_labels).float().cuda()) #.set_shape([1, 1, None, None])
    rpn_bbox_targets = Variable(torch.from_numpy(rpn_bbox_targets).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_inside_weights = Variable(torch.from_numpy(rpn_bbox_inside_weights).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_outside_weights = Variable(torch.from_numpy(rpn_bbox_outside_weights).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])

    rpn_labels = rpn_labels.long()
    self._anchor_targets['rpn_labels'] = rpn_labels
    self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
    self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
    self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

    for k in self._anchor_targets.keys():
      self._score_summaries[k] = self._anchor_targets[k]

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores):
    rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, mask_targets = \
      proposal_target_layer(
      rois, roi_scores, self._gt_boxes, self._gt_masks, self._num_classes)

    self._proposal_targets['rois'] = rois
    self._proposal_targets['labels'] = labels.long()
    self._proposal_targets['bbox_targets'] = bbox_targets
    self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
    self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
    self._proposal_targets['mask_targets'] = mask_targets

    for k in self._proposal_targets.keys():
      self._score_summaries[k] = self._proposal_targets[k]

    return rois, roi_scores

  def _anchor_component(self, height, width):
    # just to get the shape right
    #height = int(math.ceil(self._im_info.data[0, 0] / self._feat_stride[0]))
    #width = int(math.ceil(self._im_info.data[0, 1] / self._feat_stride[0]))
    anchors, anchor_length = generate_anchors_pre(\
                                          height, width,
                                           self._feat_stride, self._anchor_scales, self._anchor_ratios)
    #print('anchors:', anchors, anchors.shape)
    #[[  -38.   -16.    53.    31.]
    # [  -84.   -40.    99.    55.]
    # [ -176.   -88.   191.   103.]
    # ...,
    # [  860.   512.   947.   687.]
    # [  816.   424.   991.   775.]
    # [  728.   248.  1079.   951.]] (25992, 4)
     
    #[[ -38.  -16.   53.   31.]
    # [ -84.  -40.   99.   55.]
    # [-176.  -88.  191.  103.]
    # ...,
    # [ 748.  512.  835.  687.]
    # [ 704.  424.  879.  775.]
    # [ 616.  248.  967.  951.]] (22800, 4)
    #print('anchor_length:', anchor_length) # 25992 22800
    
    self._anchors = Variable(torch.from_numpy(anchors).cuda())
    self._anchor_length = anchor_length

  #######################
  # network construction
  #######################
  def _region_proposal(self, net_conv):
    rpn = F.relu(self.rpn_net(net_conv))
    self._act_summaries['rpn'] = rpn

    rpn_cls_score = self.rpn_cls_score_net(rpn) # batch * (num_anchors * 2) * h * w

    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = rpn_cls_score.view(self._batch_size, 2, -1, rpn_cls_score.size()[-1]) # batch * 2 * (num_anchors*h) * w
    rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)  # batch * 2 * (num_anchors*h) * w  1x2x456x57 1x2x456x50 1x2x600x38 1x2x636x38
    
    # Move channel to the last dimenstion, to fit the input of python functions
    rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
    rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
    rpn_cls_score_reshape = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous()  # batch * (num_anchors*h) * w * 2
    rpn_cls_pred = torch.max(rpn_cls_score_reshape.view(-1, 2), 1)[1]  # (batch*num_anchors*h*w, )

    rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
    rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()  # batch * h * w * (num_anchors*4)

    if self._mode == 'TRAIN':
      # produce targets and labels first
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred) # rois, roi_scores are varible
      rpn_labels = self._anchor_target_layer(rpn_cls_score)
      # generate proposals (like testing time)
      rois, _ = self._proposal_target_layer(rois, roi_scores)
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred)
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois  # (num_rois, 5) [0xyxy]

    return rois

  def _region_classification(self, spatial_fc7):
    fc7 = spatial_fc7.mean(3).mean(2) # average pooling -> (n, 2048)  (128L, 2048L)
    cls_score = self.cls_score_net(fc7) # (n, 81)  (128L, 81L)
    cls_pred = torch.max(cls_score, 1)[1] # (128L,)
    cls_prob = F.softmax(cls_score, 1) # (128L, 81L)
    bbox_pred = self.bbox_pred_net(fc7) # (128L, 324L)
    #print('--------------------')
    #print('fc7:', fc7.size())
    #print('cls_score:', cls_score.size())
    #print('cls_pred:', cls_pred.size())
    #print('cls_prob:', cls_prob.size())
    #print('bbox_pred:', bbox_pred.size())

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred

    return cls_prob, bbox_pred

  def _mask_prediction(self, spatial_fc7):
    """
    Arguments:
    - spatial_fc7 : (num_rois, 2048, 7, 7)
    Return:
    - mask_prob   : (num_rois, 80, 14, 14) range [0, 1]
    """
    upsampled_m = self.mask_up_sampling(spatial_fc7)  # (n, 256, 14, 14)  (2L, 256L, 14L, 14L) (18L, 256L, 14L, 14L)
    #print('+++++++++++++++++++++')
    #print('upsampled_m:', upsampled_m.size())
    upsampled_m = F.relu(upsampled_m) # (2L, 256L, 14L, 14L) (18L, 256L, 14L, 14L)
    mask_score = self.mask_pred_net(upsampled_m) # (n, 80, 14, 14)  (2L, 81L, 14L, 14L) (18L, 81L, 14L, 14L)
    mask_prob  = F.sigmoid(mask_score)           # (n, 80, 14, 14) range (0,1)  (2L, 81L, 14L, 14L) (18L, 81L, 14L, 14L)
    #print('upsampled_m:', upsampled_m.size())
    #print('mask_score:', mask_score.size())
    #print('mask_prob:', mask_prob.size())
    
    self._predictions['mask_score'] = mask_score 
    self._predictions['mask_prob'] = mask_prob

    return mask_prob

  def _image_to_head(self):
    raise NotImplementedError

  def _head_to_tail(self, pool5):
    raise NotImplementedError

  def create_architecture(self, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    self._tag = tag

    self._num_classes = num_classes
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    assert tag != None

    # Initialize layers
    self._init_modules()

  def init_weights(self):
    def normal_init(m, mean, stddev, truncated=False):
      """
      weight initalizer: truncated normal and random normal.
      """
      # x is a parameter
      if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
      else:
        m.weight.data.normal_(mean, stddev)
      m.bias.data.zero_()
      
    # rpn
    normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    # box
    normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)
    # mask
    normal_init(self.mask_up_sampling, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.mask_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)

  #######################
  # loss functions
  #######################
  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

  def _add_losses(self, sigma_rpn=3.0):
    # RPN, class loss
    rpn_cls_score = self._predictions['rpn_cls_score_reshape'].view(-1, 2)
    rpn_label = self._anchor_targets['rpn_labels'].view(-1)
    rpn_select = Variable((rpn_label.data != -1).nonzero().view(-1))
    rpn_cls_score = rpn_cls_score.index_select(0, rpn_select).contiguous().view(-1, 2)
    rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)
    rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

    # RPN, bbox loss
    rpn_bbox_pred = self._predictions['rpn_bbox_pred']
    rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
    rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
    rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
    rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

    # RCNN, class loss
    cls_score = self._predictions["cls_score"]
    label = self._proposal_targets["labels"].view(-1)     # (n, ) ranging [0, num_class)
    cross_entropy = F.cross_entropy(cls_score.view(-1, self._num_classes), label)

    # RCNN, bbox loss
    bbox_pred = self._predictions['bbox_pred']
    bbox_targets = self._proposal_targets['bbox_targets']
    bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
    bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
    loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # MASK, mask loss, only regress fg rois
    mask_targets = self._proposal_targets['mask_targets'] # (num_fg, 14, 14)
    mask_score   = self._predictions['mask_score']        # (num_fg, num_classes, 14, 14)
    assert mask_targets.size(0) == mask_score.size(0)
    num_fg = mask_targets.size(0)
    fg_label = label[:num_fg]  # (num_fg, )
    fg_label = fg_label.view(num_fg, 1, 1, 1).expand(num_fg, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)
    mask_score = torch.gather(mask_score, 1, fg_label) # (num_fg, 1, 14, 14)
    mask_score = mask_score.squeeze(1) # (num_fg, 14, 14)
    loss_mask = F.binary_cross_entropy_with_logits(mask_score, mask_targets)

    self._losses['cross_entropy'] = cross_entropy
    self._losses['loss_box'] = loss_box
    self._losses['rpn_cross_entropy'] = rpn_cross_entropy
    self._losses['rpn_loss_box'] = rpn_loss_box
    self._losses['loss_mask'] = loss_mask

    loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + loss_mask
    self._losses['total_loss'] = loss

    for k in self._losses.keys():
      self._event_summaries[k] = self._losses[k]

    return loss

  #######################
  # forward and backward
  #######################
  def _run_summary_op(self, val=False):
    """
    Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
    """
    summaries = []
    # Add image gt
    summaries.append(self._add_gt_image_summary())
    # Add event_summaries
    for key, var in self._event_summaries.items():
      summaries.append(tb.summary.scalar(key, var.data[0]))
    self._event_summaries = {}
    if not val:
      # Add score summaries
      for key, var in self._score_summaries.items():
        summaries.append(self._add_score_summary(key, var))
      self._score_summaries = {}
      # Add act summaries
      for key, var in self._act_summaries.items():
        summaries += self._add_act_summary(key, var)
      self._act_summaries = {}
      # Add train summaries
      for k, var in dict(self.named_parameters()).items():
        if var.requires_grad:
          summaries.append(self._add_train_summary(k, var))

      self._image_gt_summaries = {}
    
    return summaries

  def _predict(self):
    """Return
    - net_conv : (1, 1024, H, W)
    - rois     : (n, 5) [0xyxy]
    - cls_prob : (n, num_class)
    - bbox_pred: (n, num_class*4)
    - mask_prob: (num_fg, num_classes, 14, 14) eval or (n, num_classes, 14, 14) train
    """
    # This is just _build_network in tf-faster-rcnn
    torch.backends.cudnn.benchmark = False
    net_conv = self._image_to_head()

    # build the anchors for the image
    self._anchor_component(net_conv.size(2), net_conv.size(3))
   
    rois = self._region_proposal(net_conv)
    if cfg.POOLING_MODE == 'crop':
      if cfg.POOLING_ALIGN == True:
        pool5 = self._crop_pool_layer_align(net_conv, rois, self._im_info)
      else:
        pool5 = self._crop_pool_layer(net_conv, rois) # (128L, 1024L, 7L, 7L)
    else:
      pool5 = self._roi_pool_layer(net_conv, rois)

    if self._mode == 'TRAIN':
      torch.backends.cudnn.benchmark = True # benchmark because now the input size are fixed
    
    spatial_fc7 = self._head_to_tail(pool5)  # (num_rois, 2048, 7, 7)
    cls_prob, bbox_pred = self._region_classification(spatial_fc7) # (128L, 81L) (128L, 324L)

    if self._mode == 'TRAIN':
      # we only run mask prediction on foreground regions
      num_fg = self._proposal_targets['mask_targets'].size(0)
      spatial_fc7 = spatial_fc7[:num_fg]
      mask_prob = self._mask_prediction(spatial_fc7)  # (num_fg, num_classes, 14, 14)  (2L, 81L, 14L, 14L) (18L, 81L, 14L, 14L)
    else:
      mask_prob = self._mask_prediction(spatial_fc7)  # (num_rois, num_classes, 14, 14)
    
    for k in self._predictions.keys():
      self._score_summaries[k] = self._predictions[k]

    #print('net_conv:', net_conv) # (1L, 1024L, 38L, 57L) (1L, 1024L, 38L, 50L)
    #print('rois:', rois) # (128L, 5L)
    #0.0000  504.2169  370.4819  646.9880  428.3133
    #0.0000  213.2530  368.6747  540.3615  404.8193
    #0.0000  475.4428  252.3640  676.9219  401.1796
    #                    ...            
    #0.0000  287.0132  259.7431  537.6307  523.4839
    #0.0000  505.2304  241.4447  664.7133  455.1302
    #0.0000  337.4800  339.3150  533.8777  531.1620
    
    #print('cls_prob:', cls_prob) # (128L, 81L)
    #1.00000e-02 *
    #0.6753  0.8903  0.8970  ...   1.4103  0.7184  1.3966
    #1.0702  0.9341  1.0000  ...   1.9763  0.8102  1.5497
    #1.0410  0.7920  1.2093  ...   1.5631  0.9074  1.4200
    #         ...                           ...
    #1.1523  0.9095  1.1905  ...   1.9709  0.8717  1.2560
    #1.0213  0.8384  0.8523  ...   1.5380  0.9702  1.0874
    #1.1758  0.9515  1.3421  ...   1.8110  0.8519  1.1154
    
    #print('bbox_pred:', bbox_pred) # (128L, 324L)
    #6.8896e-02 -4.6045e-02  1.7904e-02  ...   2.2214e-02  1.8766e-02  5.4139e-02
    #7.2330e-02 -4.3477e-02  2.1949e-02  ...   1.7976e-04 -3.7458e-02  2.3138e-02
    #7.3659e-02 -3.1410e-02  2.0024e-02  ...  -5.9203e-03  2.8433e-02  5.1343e-02
    #               ...                                       ...
    #5.3145e-02 -5.2809e-02  2.1457e-02  ...   2.4006e-02  1.8985e-03  2.2509e-02
    #6.9023e-02 -4.5625e-02  3.2321e-02  ...   3.8104e-03  3.2275e-02  5.5593e-02
    #4.2871e-02 -7.5218e-02  2.7582e-02  ...   3.5822e-02 -5.8641e-03  2.4304e-02
    
    #print('mask_prob:', mask_prob) # (2L, 81L, 14L, 14L) (18L, 81L, 14L, 14L)
    return net_conv, rois, cls_prob, bbox_pred, mask_prob

  def _predict_masks_from_boxes_and_labels(self, net_conv, boxes, labels):
    """
    Arguments:
    - net_conv : Variable cuda (1, 1024, H, W)
    - boxes    : ndarray (n, 4) in scaled image [xyxy]
    - labels   : ndarray (n, )
    Return
    - masks    : Variable cuda (n, 14, 14), ranging [0,1]
    """
    assert self._mode == 'TEST', 'only support testing mode'

    num_boxes = boxes.shape[0]
    rois = np.hstack([np.zeros((num_boxes, 1)), boxes]).astype(np.float32) # [0xyxy] 
    rois = Variable(torch.from_numpy(rois).cuda(), volatile=True)
    if cfg.POOLING_MODE == 'crop':
      if cfg.POOLING_ALIGN == True:
        pool5 = self._crop_pool_layer_align(net_conv, rois, self._im_info)
      else:
        pool5 = self._crop_pool_layer(net_conv, rois)
    else:
      pool5 = self._roi_pool_layer(net_conv, rois)

    spatial_fc7 = self._head_to_tail(pool5) 
    mask_prob = self._mask_prediction(spatial_fc7) # (n, num_classes, 14, 14)

    # get masks from labels
    labels = Variable(torch.from_numpy(labels).long().cuda(), volatile=True)
    labels = labels.view(num_boxes, 1, 1, 1).expand(num_boxes, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)
    mask_prob = torch.gather(mask_prob, 1, labels)  # (num_boxes, 1, 14, 14)
    mask_prob = mask_prob.squeeze(1) # (num_boxes, 14, 14)

    return mask_prob

  def forward(self, image, im_info, gt_boxes=None, gt_masks=None, mode='TRAIN'):
    #image: -122.772 152.02, shape: (1, 600, 904, 3)
    #im_info: [[ 600. 904. 1.80722892]], shape: (1, 3)
    #gt_boxes:
    #[[ 504.21685791  370.48193359  646.98797607  428.31326294   65. ]
    # [ 213.25300598  368.67471313  540.3614502   404.8192749    67. ]], shape: (2, 5)
    #gt_masks: {0 1}, shape: (2, 600, 904)
    
    #image: -122.772 152.02, shape: (1, 600, 800, 3)
    #im_info: [[ 600. 800. 1.25]], shape: (1, 3)
    #gt_boxes:
    #[[  86.25   97.5   727.5   550.      6. ]
    # [ 705.    201.25  798.75  448.75    6. ]], shape: (2, 5)
    #gt_masks: {0 1}, shape: (2, 600, 800)
    
    #image: -118.672 149.47, shape: (1, 800, 600, 3)
    #im_info: [[ 800.    600.      1.25]], shape: (1, 3)
    #gt_boxes:
    #[[ 103.75  242.5   433.75  656.25   42.  ]
    # [   0.    235.    598.75  430.     67.  ]], shape: (2, 5)
    #gt_masks: {0 1}, shape: (2, 800, 600)
    
    #image: -122.748 152.003, shape: (1, 838, 600, 3)
    #im_info: [[ 838.          600.            1.31004369]], shape: (1, 3)
    #gt_boxes:
    #[[ 417.90393066  182.09606934  475.5458374   239.73799133   33.        ]
    # [  86.462883    170.30567932  302.62008667  745.41485596    1.        ]
    # [ 234.49781799  264.6288147   471.61572266  377.29257202   35.        ]], shape: (3, 5)
    #gt_masks: {0 1}, shape: (3, 838, 600)
    
    
    #print('image:', np.min(image), np.max(image), image.shape)
    #print('im_info:', im_info, im_info.shape)
    #print('gt_boxes:')
    #print(gt_boxes, gt_boxes.shape)
    #print('gt_masks:', np.unique(gt_masks), gt_masks.shape)

    self._image_gt_summaries['image'] = image
    self._image_gt_summaries['gt_boxes'] = gt_boxes
    self._image_gt_summaries['im_info'] = im_info

    self._image = Variable(torch.from_numpy(image.transpose([0,3,1,2])).cuda(), volatile=mode == 'TEST')
    self._im_info = im_info # No need to change; actually it can be a list
    self._gt_boxes = Variable(torch.from_numpy(gt_boxes).cuda()) if gt_boxes is not None else None
    self._gt_masks = gt_masks # ndarray uint8 (num_boxes, im_height, im_width), range {0,1}

    self._mode = mode

    net_conv, rois, cls_prob, bbox_pred, mask_prob = self._predict()
    #print('----------------')
    #print('net_conv:', net_conv.size()) # (1L, 1024L, 38L, 57L) (1L, 1024L, 38L, 50L)
    #print('rois:', rois.size()) # (128L, 5L)
    #print('cls_prob:', cls_prob.size()) # (128L, 81L)
    #print('bbox_pred:', bbox_pred.size()) # (128L, 324L)
    #print('mask_prob:', mask_prob.size()) # (2L, 81L, 14L, 14L) (18L, 81L, 14L, 14L)
    
    """Return
    - net_conv : (1, 1024, H, W)
    - rois     : (n, 5) [0xyxy]
    - cls_prob : (n, num_class)
    - bbox_pred: (n, num_class*4)
    - mask_prob: (num_fg, num_classes, 14, 14) eval or (n, num_classes, 14, 14) train
    """

    self._predictions['net_conv'] = net_conv

    if mode == 'TEST':   
      if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
        means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
        self._predictions["bbox_pred"] = bbox_pred.mul(Variable(stds)).add(Variable(means))
      else:
        self._predictions["bbox_pred"] = bbox_pred
    else:
      self._add_losses() # compute losses

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, image):
    feat = self._layers["head"](Variable(torch.from_numpy(image.transpose([0,3,1,2])).cuda(), volatile=True))
    return feat

  # only useful during testing mode
  def test_image(self, image, im_info):
    """Return 
    - cls_score : ndarray float32 (n, num_classes)
    - cls_prob  : ndarray float32 (n, num_classes)
    - bbox_pred : ndarray float32 (n, num_classes * 4)
    - rois      : ndarray float32 (n, 5) [0xyxy]
    - net_conv  : Variable cuda (1, 1024, H, W)
    """
    self.eval()
    self.forward(image, im_info, None, None, mode='TEST')
    cls_score, cls_prob, bbox_pred, rois, net_conv = self._predictions["cls_score"].data.cpu().numpy(), \
                                                     self._predictions['cls_prob'].data.cpu().numpy(), \
                                                     self._predictions['bbox_pred'].data.cpu().numpy(), \
                                                     self._predictions['rois'].data.cpu().numpy(), \
                                                     self._predictions['net_conv']
    return cls_score, cls_prob, bbox_pred, rois, net_conv

  def delete_intermediate_states(self):
    # Delete intermediate result to save memory
    for d in [self._losses, self._predictions, self._anchor_targets, self._proposal_targets]:
      for k in d.keys():
        del d[k]

  def get_summary(self, blobs):
    self.eval()
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['gt_masks'])
    self.train()
    summary = self._run_summary_op(True)

    return summary

  def train_step(self, blobs, train_op):
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['gt_masks'])
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_mask, loss = self._losses["rpn_cross_entropy"].data[0], \
                                                                        self._losses['rpn_loss_box'].data[0], \
                                                                        self._losses['cross_entropy'].data[0], \
                                                                        self._losses['loss_box'].data[0], \
                                                                        self._losses['loss_mask'].data[0], \
                                                                        self._losses['total_loss'].data[0]
    #utils.timer.timer.tic('backward')
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    #utils.timer.timer.toc('backward')
    train_op.step()

    self.delete_intermediate_states()

    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_mask, loss

  def train_step_with_summary(self, blobs, train_op):
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['gt_masks'])
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_mask, loss = self._losses["rpn_cross_entropy"].data[0], \
                                                                        self._losses['rpn_loss_box'].data[0], \
                                                                        self._losses['cross_entropy'].data[0], \
                                                                        self._losses['loss_box'].data[0], \
                                                                        self._losses['loss_mask'].data[0], \
                                                                        self._losses['total_loss'].data[0]
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    summary = self._run_summary_op()

    self.delete_intermediate_states()

    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_mask, loss, summary

  def train_step_no_return(self, blobs, train_op):
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['gt_masks'])
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    self.delete_intermediate_states()
    

  # def _predict_masks_from_boxes(self, net_conv, boxes):
  #   """
  #   Arguments:
  #   - net_conv : Variable cuda (1, 1024, H, W)
  #   - boxes    : ndarray (n, num_classes * 4) in scaled image
  #   Return
  #   - masks    : ndarray (n, num_classes, 14, 14) 
  #   """
  #   num_boxes = boxes.shape[0]
  #   boxes = Variable(torch.from_numpy(boxes).cuda())
  #   boxes = boxes.view(num_boxes*self._num_classes, 4) # (NC, 4)
  #   rois = torch.cat([Variable(boxes.data.new(num_boxes*self._num_classes, 1).zero_()), 
  #                     boxes], 1) # (NC, 5) [0xyxy]
  #   rois = rois.view(num_boxes, self._num_classes, 5) # (N, C, 5)

  #   masks = []  # list of num_classes (N, 14, 14) mask
  #   for c in range(self._num_classes):
  #     cth_rois = rois[:,c,:]  # (N, 5)

  #     if cfg.POOLING_MODE == 'crop':
  #       cth_pool5 = self._crop_pool_layer(net_conv, cth_rois)
  #     else:
  #       cth_pool5 = self._roi_pool_layer(net_conv, cth_rois)
      
  #     cth_spatial_fc7 = self._head_to_tail(cth_pool5)    # (N, 2048, 7, 7)
  #     mask_prob = self._mask_prediction(cth_spatial_fc7) # (N, C, 14, 14)
  #     cth_mask_prob = mask_prob[:,c,:,:] # (N, 14, 14)
  #     masks += [cth_mask_prob]
  #   masks = torch.cat([m.unsqueeze(1) for m in masks], 1)  # (N, C, 14, 14)

  #   return masks
