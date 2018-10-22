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
#from layers.joint_match import JointMatching
from layers.lang_encoder import RNNEncoder
import models.utils as model_utils
import models.eval_easy_utils as eval_utils
from crits.max_margin_crit import MaxMarginCriterion
from opt import parse_opt

# mrcn path
from model.train_val_1 import get_training_roidb, train_net
from datasets.factory import get_imdb
from model.config import cfg, cfg_from_file, cfg_from_list
import datasets.imdb
from nets.resnet_v1 import resnetv1

# torch
import torch 
import torch.nn as nn
from torch.autograd import Variable

# train one iter
def lossFun(loader, optimizer, model, mm_crit, att_crit, opt, iter):
  # set mode
  model.train()

  # zero gradient
  optimizer.zero_grad()

  # time
  T = {}

  # load one batch of data
  tic = time.time()
  data = loader.getBatch('train', opt)
  #Feats = data['Feats']
  
  image = data['image']
  im_info = data['im_info']
  gt_boxes = data['gt_boxes']
  gt_masks = data['gt_masks']
  labels = data['labels']
  print('image:', np.min(image), np.max(image), image.shape)
  print('im_info:', im_info, im_info.shape)
  print('gt_boxes:', gt_boxes, gt_boxes.shape)
  print('gt_masks:', np.unique(gt_masks), gt_masks.shape)
  print('labels:', labels, labels.shape)
  print('------------')
  
  # add [neg_vis, neg_lang]
  
  if opt['visual_rank_weight'] > 0:
    Feats = loader.combine_feats(Feats, data['neg_Feats'])
    labels = torch.cat([labels, data['labels']])
  if opt['lang_rank_weight'] > 0:
    Feats = loader.combine_feats(Feats, data['Feats'])
    labels = torch.cat([labels, data['neg_labels']])

  #att_labels, select_ixs = data['att_labels'], data['select_ixs']

  T['data'] = time.time()-tic

  # forward
  tic = time.time()
  #scores, _, sub_attn, loc_attn, rel_attn, _, _, att_scores = model(Feats['pool5'], Feats['fc7'], 
  #                                       Feats['lfeats'], Feats['dif_lfeats'], 
  #                                       Feats['cxt_fc7'], Feats['cxt_lfeats'],
  #                                       labels)
  
  output, hidden, embedded = model(labels)
  
  #loss = mm_crit(scores)
  
  #if select_ixs.numel() > 0:
  #  loss += opt['att_weight'] * att_crit(att_scores.index_select(0, select_ixs),
  #                                       att_labels.index_select(0, select_ixs))
  # if iter < 500:
  #   num_pos = len(data['ref_ids'])
  #   loss += 0.1*model.sub_rel_kl(sub_attn[:num_pos], rel_attn[:num_pos], labels[:num_pos])

  #loss.backward()
  model_utils.clip_gradient(optimizer, opt['grad_clip'])
  #optimizer.step()
  T['model'] = time.time()-tic

  # return 
  #return loss.data[0], T, data['bounds']['wrapped']
  
  return output, hidden, embedded, T, data['bounds']['wrapped']

def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb



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
  #data_json = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.json')
  #data_h5 = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.h5')
  
  #loader = GtMRCNLoader(data_json, data_h5)
  
  # prepare feats
  #feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
  #head_feats_dir=osp.join('cache/feats', opt['dataset_splitBy'], 'mrcn', feats_dir)
  #loader.prepare_mrcn(head_feats_dir, args)
  #ann_feats = osp.join('cache/feats', opt['dataset_splitBy'], 'mrcn', 
  #                     '%s_%s_%s_ann_feats.h5' % (opt['net_name'], opt['imdb_name'], opt['tag']))
  #loader.loadFeats({'ann': ann_feats})
  
  # set up model
  #opt['vocab_size']= loader.vocab_size
  #opt['fc7_dim']   = loader.fc7_dim
  #opt['pool5_dim'] = loader.pool5_dim
  #opt['num_atts']  = loader.num_atts
  #model = JointMatching(opt)
  
  net = resnetv1(batch_size=1, num_layers=101)
  
  # train set
  imdb, roidb = combined_roidb('coco_2014_train_minus_refer_valtest+coco_2014_valminusminival')
  #imdb, roidb = combined_roidb('refcoco_train+refcoco_val')
  print('{:d} roidb entries'.format(len(roidb)))
  

  # output directory where the models are saved
  output_dir = 'output_tmp'
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = 'tb_tmp'
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb('coco_2014_minival')
  #_, valroidb = combined_roidb('refcoco_test')
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip
  
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)
  
  train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
            pretrained_model='/4TB/ywchen/lang2seg/pyutils/mask-faster-rcnn/output/res101/coco_2014_train_minus_refer_valtest+coco_2014_valminusminival/notime/res101_mask_rcnn_iter_1250000.pth',
            max_iters=args.max_iters)
  
  """
  LSTM = RNNEncoder(vocab_size=opt['vocab_size'],
                    word_embedding_size=opt['word_embedding_size'],
                    word_vec_size=opt['word_vec_size'],
                    hidden_size=opt['rnn_hidden_size'],
                    bidirectional=opt['bidirectional']>0,
                    input_dropout_p=opt['word_drop_out'],
                    dropout_p=opt['rnn_drop_out'],
                    n_layers=opt['rnn_num_layers'],
                    rnn_type=opt['rnn_type'],
                    variable_lengths=opt['variable_lengths']>0)
  
  # resume from previous checkpoint
  infos = {}
  if opt['start_from'] is not None:
    pass
  iter = infos.get('iter', 0)
  epoch = infos.get('epoch', 0)
  val_accuracies = infos.get('val_accuracies', [])
  val_loss_history = infos.get('val_loss_history', {})
  val_result_history = infos.get('val_result_history', {})
  loss_history = infos.get('loss_history', {})
  loader.iterators = infos.get('iterators', loader.iterators)
  if opt['load_best_score'] == 1:
    best_val_score = infos.get('best_val_score', None)

  # set up criterion
  mm_crit = MaxMarginCriterion(opt['visual_rank_weight'], opt['lang_rank_weight'], opt['margin'])
  att_crit = nn.BCEWithLogitsLoss(loader.get_attribute_weights())

  # move to GPU
  if opt['gpuid'] >= 0:
    #model.cuda()
    LSTM.cuda()
    mm_crit.cuda()
    att_crit.cuda()

  # set up optimizer
  optimizer = torch.optim.Adam(LSTM.parameters(), 
                               lr=opt['learning_rate'],
                               betas=(opt['optim_alpha'], opt['optim_beta']),
                               eps=opt['optim_epsilon'])

  # start training
  data_time, model_time = 0, 0
  lr = opt['learning_rate']
  best_predictions, best_overall = None, None
  while True:
    # run one iteration
    #loss, T, wrapped = lossFun(loader, optimizer, model, mm_crit, att_crit, opt, iter)
    
    output, hidden, embedded, T, wrapped = lossFun(loader, optimizer, LSTM, mm_crit, att_crit, opt, iter)
    
    data_time += T['data']
    model_time += T['model']
    
    print('--------')
    print(output.size())
    print(hidden.size())
    print(embedded.size())
    print('********')
    
    # write the training loss summary
    if iter % opt['losses_log_every'] == 0:
      loss_history[iter] = loss
      # print stats
      log_toc = time.time()
      print('iter[%s](epoch[%s]), train_loss=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter' \
            % (iter, epoch, loss, lr, data_time/opt['losses_log_every'], model_time/opt['losses_log_every']))
      data_time, model_time = 0, 0

    # decay the learning rates
    if opt['learning_rate_decay_start'] > 0 and iter > opt['learning_rate_decay_start']:
      frac = (iter - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
      decay_factor =  0.1 ** frac
      lr = opt['learning_rate'] * decay_factor
      # update optimizer's learning rate
      model_utils.set_lr(optimizer, lr)

    # eval loss and save checkpoint
    if iter % opt['save_checkpoint_every'] == 0 or iter == opt['max_iters']:
      val_loss, acc, predictions, overall = eval_utils.eval_split(loader, model, None, 'val', opt)
      val_loss_history[iter] = val_loss
      val_result_history[iter] = {'loss': val_loss, 'accuracy': acc}
      val_accuracies += [(iter, acc)]
      print('validation loss: %.2f' % val_loss)
      print('validation acc : %.2f%%\n' % (acc*100.0))
      print('validation precision : %.2f%%' % (overall['precision']*100.0))
      print('validation recall    : %.2f%%' % (overall['recall']*100.0))
      print('validation f1        : %.2f%%' % (overall['f1']*100.0))       
            
      # save model if best
      current_score = acc
      if best_val_score is None or current_score > best_val_score:
        best_val_score = current_score
        best_predictions = predictions
        best_overall = overall
        checkpoint_path = osp.join(checkpoint_dir, opt['id'] + '.pth')
        checkpoint = {}
        checkpoint['model'] = model
        checkpoint['opt'] = opt
        torch.save(checkpoint, checkpoint_path) 
        print('model saved to %s' % checkpoint_path) 

      # write json report
      infos['iter'] = iter
      infos['epoch'] = epoch
      infos['iterators'] = loader.iterators
      infos['loss_history'] = loss_history
      infos['val_accuracies'] = val_accuracies
      infos['val_loss_history'] = val_loss_history
      infos['best_val_score'] = best_val_score
      infos['best_predictions'] = predictions if best_predictions is None else best_predictions
      infos['best_overall'] = overall if best_overall is None else best_overall
      infos['opt'] = opt
      infos['val_result_history'] = val_result_history
      infos['word_to_ix'] = loader.word_to_ix
      infos['att_to_ix'] = loader.att_to_ix
      with open(osp.join(checkpoint_dir, opt['id']+'.json'), 'wb') as io:
        json.dump(infos, io)
    
    # update iter and epoch
    iter += 1
    if wrapped:
      epoch += 1
    if iter >= opt['max_iters'] and opt['max_iters'] > 0:
      break
  """

if __name__ == '__main__':

  args = parse_opt()
  main(args)

