# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.network_vgg import Network
from model.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models

from layers.lang_encoder import RNNEncoder

class vgg16(Network):
  def __init__(self, opt, batch_size=1):
    Network.__init__(self, batch_size=batch_size)

    # language rnn encoder
    self.rnn_encoder = RNNEncoder(vocab_size=opt['vocab_size'],
                                  word_embedding_size=opt['word_embedding_size'],
                                  word_vec_size=opt['word_vec_size'],
                                  hidden_size=opt['rnn_hidden_size'],
                                  bidirectional=opt['bidirectional']>0,
                                  input_dropout_p=opt['word_drop_out'],
                                  dropout_p=opt['rnn_drop_out'],
                                  n_layers=opt['rnn_num_layers'],
                                  rnn_type=opt['rnn_type'],
                                  variable_lengths=opt['variable_lengths']>0)

    self._rnn_num_layers = opt['rnn_num_layers']
    self._rnn_hidden_size = opt['rnn_hidden_size']
    self._rnn_num_dirs = 2 if opt['bidirectional'] > 0 else 1
    self._C4_feat_dim = opt['C4_feat_dim']

  def _init_modules(self):
    self.vgg = models.vgg16()
    # Remove fc8
    self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.vgg.features[layer].parameters(): p.requires_grad = False

    # not using the last maxpool layer
    self._layers['head'] = nn.Sequential(*list(self.vgg.features._modules.values())[:-1])
    
    # dynamic filter generator
    self.dynamic_fc_0 = nn.Linear(self._rnn_num_layers * self._rnn_num_dirs * self._rnn_hidden_size, self._C4_feat_dim)
    self.dynamic_fc_1 = nn.Linear(self._rnn_num_layers * self._rnn_num_dirs * self._rnn_hidden_size, self._C4_feat_dim)
    self.dynamic_fc_2 = nn.Linear(self._rnn_num_layers * self._rnn_num_dirs * self._rnn_hidden_size, self._C4_feat_dim)
    self.dynamic_fc_3 = nn.Linear(self._rnn_num_layers * self._rnn_num_dirs * self._rnn_hidden_size, self._C4_feat_dim)
    self.dynamic_fc_4 = nn.Linear(self._rnn_num_layers * self._rnn_num_dirs * self._rnn_hidden_size, self._C4_feat_dim)
    self.dynamic_fc_5 = nn.Linear(self._rnn_num_layers * self._rnn_num_dirs * self._rnn_hidden_size, self._C4_feat_dim)
    self.dynamic_fc_6 = nn.Linear(self._rnn_num_layers * self._rnn_num_dirs * self._rnn_hidden_size, self._C4_feat_dim)
    
    self.response_fc = nn.Linear(self._rnn_num_layers * self._rnn_num_dirs * self._rnn_hidden_size, 7)
    
    # rpn
    self.rpn_net = nn.Conv2d(512, 512, [3, 3], padding=1)

    self.rpn_cls_score_net = nn.Conv2d(512, self._num_anchors * 2, [1, 1])
    
    self.rpn_bbox_pred_net = nn.Conv2d(512, self._num_anchors * 4, [1, 1])

    self.cls_score_net = nn.Linear(4096, self._num_classes)
    self.bbox_pred_net = nn.Linear(4096, self._num_classes * 4)

    self.init_weights()

  def _image_to_head(self):
    net_conv = self._layers['head'](self._image)
    self._act_summaries['conv'] = net_conv
    
    return net_conv

  def _head_to_tail(self, pool5):
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.vgg.classifier(pool5_flat)

    return fc7

  def load_pretrained_cnn(self, state_dict):
    self.vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()})