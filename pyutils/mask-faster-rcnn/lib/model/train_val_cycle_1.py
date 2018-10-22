# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorboardX as tb

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
import utils.timer
try:
  import cPickle as pickle
except ImportError:
  import pickle

import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import os
import sys
import glob
import time
import random

def scale_lr(optimizer, scale):
  """Scale the learning rate of the optimizer"""
  for param_group in optimizer.param_groups:
    param_group['lr'] *= scale

class SolverWrapper(object):
  """
    A wrapper class for the training process
  """

  #def __init__(self, network, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None):
  def __init__(self, network, loader, output_dir, tbdir, pretrained_model=None):
    self.net = network
    #self.imdb = imdb
    #self.roidb = roidb
    #self.valroidb = valroidb
    self.loader = loader
    self.output_dir = output_dir
    self.tbdir = tbdir
    # Simply put '_val' at the end to save the summaries from the validation set
    self.tbvaldir = tbdir + '_val'
    if not os.path.exists(self.tbvaldir):
      os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model

  def snapshot(self, iter):
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pth'
    filename = os.path.join(self.output_dir, filename)
    torch.save(self.net.state_dict(), filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    # Also store some meta information, random state, etc.
    nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(self.output_dir, nfilename)
    # current state of numpy random
    st0 = np.random.get_state()
    st1 = random.getstate()
    iter_train = self.loader.iterators['train']
    perm_train = self.loader.perm['train']
    iter_val = self.loader.iterators['val']
    perm_val = self.loader.perm['val']
    
    # current position in the database
    #cur = self.data_layer._cur
    # current shuffled indexes of the database
    #perm = self.data_layer._perm
    # current position in the validation database
    #cur_val = self.data_layer_val._cur
    # current shuffled indexes of the validation database
    #perm_val = self.data_layer_val._perm

    # Dump the meta info
    with open(nfilename, 'wb') as fid:
      pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(st1, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter_train, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_train, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      
      #pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
      #pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
      #pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
      #pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

    return filename, nfilename

  def from_snapshot(self, sfile, nfile):
    print('Restoring model snapshots from {:s}'.format(sfile))
    #self.net.load_state_dict(torch.load(str(sfile)))
    
    ####
    saved_state_dict = torch.load(str(sfile))
    count_0 = 0
    count_1 = 0
    new_params = self.net.state_dict().copy()
    for name, param in new_params.items():
      #print(name, param.size(), saved_state_dict[name].size())
      #if name in saved_state_dict and 'rpn_net.weight' not in name and 'resnet.layer4.0.conv1.weight' not in name and 'resnet.layer4.0.downsample.0.weight' not in name:
      if name in saved_state_dict and param.size() == saved_state_dict[name].size():
        new_params[name].copy_(saved_state_dict[name])
        #print('---- copy ----')
      elif name in saved_state_dict and param[:,:-1,:,:].size() == saved_state_dict[name].size():
        new_params[name][:,:-1,:,:].copy_(saved_state_dict[name])
        print(name, param.size(), '->', param[:,:-1,:,:].size(), saved_state_dict[name].size(), '----')
        count_0 += 1
      else:
        print(name, '----')
        count_1 += 1
    print('size partially match:', count_0)
    print('size not match:', count_1)
    self.net.load_state_dict(new_params)
    ####
    
    print('Restored.')
    # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
    # tried my best to find the random states so that it can be recovered exactly
    # However the Tensorflow state is currently not available
    with open(nfile, 'rb') as fid:
      st0 = pickle.load(fid)
      st1 = pickle.load(fid)
      iter_train = pickle.load(fid)
      perm_train = pickle.load(fid) ####
      iter_val = pickle.load(fid)
      perm_val = pickle.load(fid) ####
      
      #cur = pickle.load(fid)
      #perm = pickle.load(fid)
      #cur_val = pickle.load(fid)
      #perm_val = pickle.load(fid)
      last_snapshot_iter = pickle.load(fid)

      np.random.set_state(st0)
      random.setstate(st1)
      self.loader.iterators['train'] = iter_train
      #self.loader.perm['train'] = np.arange(len(self.loader.split_ix['train'])) ####
      self.loader.perm['train'] = perm_train ####
      self.loader.iterators['val'] = iter_val
      #self.loader.perm['val'] = np.arange(len(self.loader.split_ix['val'])) ####
      self.loader.perm['val'] = perm_val ####
      
      #self.data_layer._cur = cur
      #self.data_layer._perm = perm
      #self.data_layer_val._cur = cur_val
      #self.data_layer_val._perm = perm_val

    return last_snapshot_iter

  def construct_graph(self):
    # Set the random seed
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    # Build the main computation graph
    #self.net.create_architecture(self.imdb.num_classes, tag='default',
    self.net.create_architecture(81, tag='default',
                                 anchor_scales=cfg.ANCHOR_SCALES,
                                 anchor_ratios=cfg.ANCHOR_RATIOS)
    # Define the loss
    # loss = layers['total_loss']
    # Set learning rate and momentum
    if cfg.TRAIN.FROM_FRCN:
      params = []
      for key, value in dict(self.net.named_parameters()).items():
        if value.requires_grad:
          if 'mask' in key:
            lr = cfg.TRAIN.LEARNING_RATE
            print('%s\'s lr (%.4f) will not be suppressed.' % (key, lr))
          else:
            lr = cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA # 1/10 lr
          if 'bias' in key:
            params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), 'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
          else:
            params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    else: ####
      lr = cfg.TRAIN.LEARNING_RATE
      params = []
      for key, value in dict(self.net.named_parameters()).items():
        if value.requires_grad: # layer1 (when RESNET.FIXED_BLOCKS = 1), bn, downsample excluded
          #if ('rnn_encoder' in key or 'dynamic_fc' in key or 'response' in key) and 'bias' in key:
          #  params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1)*10, 'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
          #  print(key, '++++****', lr*(cfg.TRAIN.DOUBLE_BIAS + 1)*10)
          #elif 'rnn_encoder' in key or 'dynamic_fc' in key or 'response' in key:
          #  params += [{'params':[value],'lr':lr*10, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
          #  print(key, '----****', lr*10)
            
          if 'bias' in key:
            params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), 'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            print(key, '++++++++', lr*(cfg.TRAIN.DOUBLE_BIAS + 1))
          else:
            params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
            print(key, '--------', lr)

    self.optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    # Write the train and validation information to tensorboard
    self.writer = tb.writer.FileWriter(self.tbdir)
    self.valwriter = tb.writer.FileWriter(self.tbvaldir)

    return lr, self.optimizer

  def find_previous(self):
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pth')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in pytorch
    redfiles = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      redfiles.append(os.path.join(self.output_dir, 
                      cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.pth'.format(stepsize+1)))
    sfiles = [ss for ss in sfiles if ss not in redfiles]

    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)
    redfiles = [redfile.replace('.pth', '.pkl') for redfile in redfiles]
    nfiles = [nn for nn in nfiles if nn not in redfiles]

    lsf = len(sfiles)
    assert len(nfiles) == lsf

    return lsf, nfiles, sfiles

  def initialize(self):
    # Initial file lists are empty
    np_paths = []
    ss_paths = []
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(self.pretrained_model))
    
    #if cfg.TRAIN.FROM_FRCN:
    #  self.net.load_pretrained_frcn(torch.load(self.pretrained_model))
    #else:
    #  self.net.load_pretrained_cnn(torch.load(self.pretrained_model))
    
    ####
    saved_state_dict = torch.load(self.pretrained_model)
    count_0 = 0
    count_1 = 0
    new_params = self.net.state_dict().copy()
    for name, param in new_params.items():
      #print(name, param.size(), saved_state_dict[name].size())
      #if name in saved_state_dict and 'rpn_net.weight' not in name and 'resnet.layer4.0.conv1.weight' not in name and 'resnet.layer4.0.downsample.0.weight' not in name:
      if name in saved_state_dict and param.size() == saved_state_dict[name].size():
        new_params[name].copy_(saved_state_dict[name])
        #print('---- copy ----')
      elif name in saved_state_dict and param[:,:-1,:,:].size() == saved_state_dict[name].size():
        new_params[name][:,:-1,:,:].copy_(saved_state_dict[name])
        print(name, param.size(), '->', param[:,:-1,:,:].size(), saved_state_dict[name].size(), '----')
        count_0 += 1
      else:
        print(name, '----')
        count_1 += 1
    print('size partially match:', count_0)
    print('size not match:', count_1)
    self.net.load_state_dict(new_params)
    ####
    
    print('Loaded.')
    # Need to fix the variables before loading, so that the RGB weights are changed to BGR
    # For VGG16 it also changes the convolutional weights fc6 and fc7 to
    # fully connected weights
    last_snapshot_iter = 0
    lr = cfg.TRAIN.LEARNING_RATE
    stepsizes = list(cfg.TRAIN.STEPSIZE)

    return lr, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def restore(self, sfile, nfile):
    # Get the most recent snapshot and restore
    np_paths = [nfile]
    ss_paths = [sfile]
    # Restore model from snapshots
    last_snapshot_iter = self.from_snapshot(sfile, nfile)
    # Set the learning rate
    lr_scale = 1
    stepsizes = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      if last_snapshot_iter > stepsize:
        lr_scale *= cfg.TRAIN.GAMMA
      else:
        stepsizes.append(stepsize)
    scale_lr(self.optimizer, lr_scale)
    lr = cfg.TRAIN.LEARNING_RATE * lr_scale
    return lr, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def remove_snapshot(self, np_paths, ss_paths):
    to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      nfile = np_paths[0]
      os.remove(str(nfile))
      np_paths.remove(nfile)

    to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      sfile = ss_paths[0]
      # To make the code compatible to earlier versions of Tensorflow,
      # where the naming tradition for checkpoints are different
      os.remove(str(sfile))
      ss_paths.remove(sfile)

  def train_model(self, max_iters):
    # Build data layers for both training and validation set
    #self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    #self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)

    # Construct the computation graph
    lr, train_op = self.construct_graph()

    # Find previous snapshots if there is any to restore from
    lsf, nfiles, sfiles = self.find_previous()

    # Initialize the variables or restore them from the last snapshot
    if lsf == 0:
      lr, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize()
    else:
      lr, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(str(sfiles[-1]), 
                                                                             str(nfiles[-1]))
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()
    # Make sure the lists are not empty
    stepsizes.append(max_iters)
    stepsizes.reverse()
    next_stepsize = stepsizes.pop()

    self.net.train()
    self.net.cuda()

    print('perm_train:', self.loader.perm['train'])
    print('perm_val:', self.loader.perm['val'])
    while iter < max_iters + 1:
      #torch.cuda.empty_cache()

      # Get training data, one batch at a time
      #blobs = self.data_layer.forward()
      blobs = self.loader.getBatch('train', self.net._batch_size)
      sent_num = blobs['gt_boxes'].shape[0]
      arr = np.random.permutation(sent_num)
      #arr = np.arange(sent_num)
      #np.random.shuffle(arr)
      #print('sent_num:', sent_num)
      #print('arr:', arr)

      for idx in range(sent_num):
        utils.timer.timer.tic()
        # Learning rate
        if iter == next_stepsize + 1:
          # Add snapshot here before reducing the learning rate
          self.snapshot(iter)
          lr *= cfg.TRAIN.GAMMA
          scale_lr(self.optimizer, cfg.TRAIN.GAMMA)
          next_stepsize = stepsizes.pop()
        
        #print('iter:', iter, 'arr[idx]:', arr[idx])
        now = time.time()
        if iter == 1 or iter % 500 == 0: #now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL: ####
          print('----------- summary -----------')
          # Compute the graph with summary
          rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_mask, loss_caption, total_loss, summary = \
            self.net.train_step_with_summary(blobs, arr[idx], self.optimizer)
          for _sum in summary: self.writer.add_summary(_sum, float(iter))
          # Also check the summary on the validation set
          #blobs_val = self.data_layer_val.forward() ####
          blobs_val = self.loader.getBatch('val')
          idx_val = random.randint(0, blobs_val['gt_boxes'].shape[0]-1)
          summary_val = self.net.get_summary(blobs_val, idx_val)
          for _sum in summary_val: self.valwriter.add_summary(_sum, float(iter))
          last_summary_time = now
          
          print('------------------------------------------------')
          for key, value in dict(self.net.named_parameters()).items():
            if 'resnet.fc' in key:
              print(key, value)
          
        else:
          # Compute the graph without summary
          rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_mask, loss_caption, total_loss = \
            self.net.train_step(blobs, arr[idx], self.optimizer)
        utils.timer.timer.toc()
        

        
        

        # Display training information
        if iter % (cfg.TRAIN.DISPLAY) == 0:
          print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> loss_mask: %.6f\n >>> loss_caption: %.6f\n >>> lr: %f' % \
                (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_mask, loss_caption, lr))
          print('speed: {:.3f}s / iter'.format(utils.timer.timer.average_time()))

          # for k in utils.timer.timer._average_time.keys():
          #   print(k, utils.timer.timer.average_time(k))

        # Snapshotting
        if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
          last_snapshot_iter = iter
          ss_path, np_path = self.snapshot(iter)
          np_paths.append(np_path)
          ss_paths.append(ss_path)

          # Remove the old snapshots if there are too many
          if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
            self.remove_snapshot(np_paths, ss_paths)

        #print('iter:', iter, 'iterators:', self.loader.iterators['train'], self.loader.iterators['val'])
        iter += 1

    if last_snapshot_iter != iter - 1:
      self.snapshot(iter - 1)

    self.writer.close()
    self.valwriter.close()


def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb


def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""

  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


#def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
def train_net(network, loader, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000):
  """Train a Faster R-CNN network."""
  #roidb = filter_roidb(roidb)
  #valroidb = filter_roidb(valroidb)

  #sw = SolverWrapper(network, imdb, roidb, valroidb, output_dir, tb_dir,
  sw = SolverWrapper(network, loader, output_dir, tb_dir,
                     pretrained_model=pretrained_model)

  print('Solving...')
  sw.train_model(max_iters)
  print('done solving')
