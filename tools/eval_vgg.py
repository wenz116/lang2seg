from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import json
import numpy as np
import h5py
import time
from pprint import pprint
import argparse
import pprint

# model
import _init_paths
from loaders.gt_mrcn_loader import GtMRCNLoader
import models.eval_dets_utils as eval_utils

from nets.vgg16 import vgg16
from model.config_vgg import cfg, cfg_from_file, cfg_from_list
from model.test_vgg import eval_split
from opt import parse_opt

# torch
import torch
import torch.nn as nn

def load_model(checkpoint_path, opt):
  tic = time.time()
  model = JointMatching(opt)
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model'].state_dict())
  model.eval()
  model.cuda()
  print('model loaded in %.2f seconds' % (time.time()-tic))
  return model

def evaluate(args):

  opt = vars(args)
  
  # make other options
  opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
  
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  # set up loader
  data_json = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.json')
  data_h5 = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.h5')
  loader = GtMRCNLoader(data_json, data_h5)
  
  # set up model
  opt['vocab_size']= loader.vocab_size
  opt['C4_feat_dim'] = 512
  net = vgg16(opt, batch_size=1)
  
  
  net.create_architecture(81, tag='default',
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS)
  
  sfile = osp.join(opt['dataset_splitBy'], 'output_{}'.format(opt['output_postfix']), 'vgg16_faster_rcnn_iter_{}.pth'.format(opt['model_iter']))
  print('Restoring model snapshots from {:s}'.format(sfile))
  saved_state_dict = torch.load(str(sfile))
  count_1 = 0
  new_params = net.state_dict().copy()
  for name, param in new_params.items():
    #print(name, param.size(), saved_state_dict[name].size())
    if name in saved_state_dict and param.size() == saved_state_dict[name].size():
      new_params[name].copy_(saved_state_dict[name])
      #print('---- copy ----')
    else:
      print(name, '----')
      count_1 += 1
  print('size not match:', count_1)
  net.load_state_dict(new_params)
  
  net.eval()
  net.cuda()
  
  split = opt['split']
  
  crit = None
  acc, num_sent = eval_split(loader, net, crit, split, opt)
  print('Comprehension on %s\'s %s (%s sents) is %.2f%%' % \
        (opt['dataset_splitBy'], split, num_sent, acc*100.))
  
  # write to results.txt
  f = open('experiments/det_results.txt', 'a')
  f.write('[%s][%s], id[%s]\'s acc is %.2f%%\n' % \
          (opt['dataset_splitBy'], opt['split'], opt['id'], acc*100.0))



  # print 
  #print('Segmentation results on [%s][%s]' % (opt['dataset_splitBy'], split))
  #results_str = ''
  #for n_eval_iou in range(len(eval_seg_iou_list)):
  #  results_str += '    precision@%s = %.2f\n' % \
  #    (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou]*100./seg_total)
  #results_str += '    overall IoU = %.2f\n' % (cum_I*100./cum_U)
  #print(results_str)

  # save results
  #save_dir = osp.join('cache/results', opt['dataset_splitBy'], 'masks')
  #if not osp.isdir(save_dir):
  #  os.makedirs(save_dir)

  #results['iou'] = cum_I*1./cum_U
  #assert 'rle' in results['predictions'][0]
  #with open(osp.join(save_dir, args.id+'_'+split+'.json'), 'w') as f:
  #  json.dump(results, f)

  # write to results.txt
  #f = open('experiments/mask_results.txt', 'a')
  #f.write('[%s][%s]\'s iou is:\n%s' % \
  #        (opt['dataset_splitBy'], split, results_str))



if __name__ == '__main__':

  #parser = argparse.ArgumentParser()
  #parser.add_argument('--dataset', type=str, default='refcoco', help='dataset name: refclef, refcoco, refcoco+, refcocog')
  #parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
  #parser.add_argument('--split', type=str, default='testA', help='split: testAB or val, etc')
  #parser.add_argument('--id', type=str, default='0', help='model id name')
  #parser.add_argument('--num_sents', type=int, default=-1, help='how many sentences to use when periodically evaluating the loss? (-1=all)')
  #parser.add_argument('--verbose', type=int, default=1, help='if we want to print the testing progress')
  #parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
  #parser.add_argument('--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)
  #args = parser.parse_args()

  args = parse_opt()
  evaluate(args)
