from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import numpy as np
import json
import h5py
import time
import random
from pprint import pprint

# model
import _init_paths
from loaders.gt_mrcn_loader import GtMRCNLoader
import models.utils as model_utils
from opt import parse_opt

# mrcn path
from model.train_val import get_training_roidb, train_net
from datasets.factory import get_imdb
from model.config import cfg, cfg_from_file, cfg_from_list
import datasets.imdb
from nets.resnet_v1_7f import resnetv1

# torch
import torch 
import torch.nn as nn
from torch.autograd import Variable


def main(args):

  opt = vars(args)

  # initialize
  opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
  checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'])
  if not osp.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)

  # set random seed
  torch.manual_seed(opt['seed'])
  random.seed(opt['seed'])
  
  # set up loader
  data_json = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.json')
  data_h5 = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.h5')
  
  loader = GtMRCNLoader(data_json, data_h5)
  
  # set up model
  opt['vocab_size']= loader.vocab_size
  opt['C4_feat_dim'] = 1024
  net = resnetv1(opt, batch_size=opt['batch_size'], num_layers=101) # determine batch size in opt.py

  # output directory where the models are saved
  output_dir = osp.join(opt['dataset_splitBy'], 'output_{}'.format(opt['output_postfix']))
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = osp.join(opt['dataset_splitBy'], 'tb_{}'.format(opt['output_postfix']))
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  cfg.TRAIN.USE_FLIPPED = orgflip
  
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)
  
  #train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
  train_net(net, loader, output_dir, tb_dir,
            pretrained_model='pyutils/mask-faster-rcnn/output/res101/coco_2014_train_minus_refer_valtest+coco_2014_valminusminival/notime/res101_mask_rcnn_iter_1250000.pth',
            max_iters=args.max_iters)


if __name__ == '__main__':

  args = parse_opt()
  main(args)
