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
from loaders.cycle_loader import CycleLoader
#from layers.joint_match import JointMatching
#from layers.lang_encoder import RNNEncoder
import models.utils as model_utils
#import models.eval_easy_utils as eval_utils
#from crits.max_margin_crit import MaxMarginCriterion
from opt_cycle import parse_opt
import misc.utils as utils

# mrcn path
from model.train_val_cycle import get_training_roidb, train_net
from datasets.factory import get_imdb
from model.config import cfg, cfg_from_file, cfg_from_list
import datasets.imdb
from nets.resnet_v1_cycle import resnetv1

# torch
import torch 
import torch.nn as nn
from torch.autograd import Variable

from six.moves import cPickle


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
  
  loader = CycleLoader(data_json, data_h5) ####
  
  # prepare feats
  #feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
  #head_feats_dir=osp.join('cache/feats', opt['dataset_splitBy'], 'mrcn', feats_dir)
  #loader.prepare_mrcn(head_feats_dir, args)
  #ann_feats = osp.join('cache/feats', opt['dataset_splitBy'], 'mrcn', 
  #                     '%s_%s_%s_ann_feats.h5' % (opt['net_name'], opt['imdb_name'], opt['tag']))
  #loader.loadFeats({'ann': ann_feats})
  
  # set up model
  opt['vocab_size']= loader.vocab_size
  #opt['fc7_dim']   = loader.fc7_dim
  #opt['pool5_dim'] = loader.pool5_dim
  #opt['num_atts']  = loader.num_atts
  #model = JointMatching(opt)
  opt['C4_feat_dim'] = 1024
  opt['use_att'] = utils.if_use_att(opt['caption_model'])
  opt['seq_length'] = loader.label_length
  
  
  #### can change to restore opt from info.pkl
  #infos = {}
  #histories = {}
  if opt['start_from']is not None:
    # open old infos and check if models are compatible
    with open(os.path.join(opt['start_from'], 'infos.pkl')) as f:
      infos = cPickle.load(f)
      saved_model_opt = infos['opt']
      need_be_same = ['caption_model', 'rnn_type', 'rnn_size', 'num_layers']
      for checkme in need_be_same:
        assert vars(saved_model_opt)[checkme] == opt[checkme], "Command line argument and saved model disagree on '%s'" % checkme

    #if os.path.isfile(os.path.join(opt['start_from'], 'histories.pkl')):
    #  with open(os.path.join(opt['start_from'], 'histories.pkl')) as f:
    #    histories = cPickle.load(f)
  
  
  
  net = resnetv1(opt, batch_size=opt['batch_size'], num_layers=101) #### determine batch size in opt.py
  
  # train set
  #imdb, roidb = combined_roidb('coco_2014_train_minus_refer_valtest+coco_2014_valminusminival')
  #imdb, roidb = combined_roidb('refcoco_train+refcoco_val')
  #print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved
  output_dir = 'output_1e-4-3_0k_cycle_0.1'
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = 'tb_1e-4-3_0k_cycle_0.1'
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  #_, valroidb = combined_roidb('coco_2014_minival')
  #_, valroidb = combined_roidb('refcoco_test')
  #print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip
  
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)
  
  
  #train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
  train_net(net, loader, output_dir, tb_dir,
            pretrained_model='/4TB/ywchen/lang2seg/pyutils/mask-faster-rcnn/output/res101/coco_2014_train_minus_refer_valtest+coco_2014_valminusminival/notime/res101_mask_rcnn_iter_1250000.pth',
            max_iters=args.max_iters)


if __name__ == '__main__':

  args = parse_opt()
  main(args)