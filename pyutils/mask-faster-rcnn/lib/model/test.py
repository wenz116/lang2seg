# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from utils.blob import im_list_to_blob
from utils.mask_utils import recover_masks, recover_cls_masks
from utils.visualization import draw_bounding_boxes

from pycocotools import mask as COCOmask

from model.nms_wrapper import nms
from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

import torch

from PIL import Image
from PIL import ImageFilter
from scipy.misc import imresize
import time

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(net, blobs):
  """Return
  - scores     : (num_rois, num_classes)
  - pred_boxes : (num_rois, num_classes * 4) [xyxy] in original image size
  - net_conv   : Variable cuda (1, 1024, H, W)
  - im_scale   : float
  """
  #blobs, im_scales = _get_blobs(im)
  #assert len(im_scales) == 1, "Only single-image batch implemented"

  #m_blob = blobs['data']
  #blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
  #blobs = self.loader.getBatch('train') ####

  _, scores, bbox_pred, rois, net_conv = net.test_image(blobs)
  #test_image(self, image, im_info, labels, file_name)
  
  boxes = rois[:, 1:5] / blobs['im_info'][0][2] # (n, 4)
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1]) # (n, C*4)
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
    im_shape = (round(blobs['im_info'][0][0]/blobs['im_info'][0][2]), round(blobs['im_info'][0][1]/blobs['im_info'][0][2]), 3) ####
    pred_boxes = _clip_boxes(pred_boxes, im_shape) # (n, C*4)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes, net_conv, blobs['im_info'][0][2]


def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[im_ind][cls_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(torch.from_numpy(dets), thresh).numpy()
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes


# IoU function
def computeIoU_box(box1, box2):
  # each box is of [x1, y1, x2, y2]
  inter_x1 = max(box1[0], box2[0])
  inter_y1 = max(box1[1], box2[1])
  inter_x2 = min(box1[2], box2[2])
  inter_y2 = min(box1[3], box2[3])

  if inter_x1 < inter_x2 and inter_y1 < inter_y2:
    inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
  else:
    inter = 0
  union = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1) + (box2[2]-box2[0]+1)*(box2[3]-box2[1]+1) - inter
  return float(inter)/union


# compute IoU
def computeIoU_seg(pred_seg, gt_seg):
  I = np.sum(np.logical_and(pred_seg, gt_seg))
  U = np.sum(np.logical_or(pred_seg, gt_seg))
  return I, U


def eval_split(loader, model, crit, split, opt, max_per_image=100, thresh=0.):
  verbose = opt.get('verbose', True)
  num_sents = opt.get('num_sents', -1)
  #assert split != 'train', 'Check the evaluation split. (comment this line if you are evaluating [train])'

  # set mode
  model.eval()

  # initialize
  n = 0
  loss_evals = 0
  acc = 0
  num_sent = 0
  #predictions = []
  finish_flag = False
  
  num_refs = {'train': 42404, 'val': 3811, 'testA': 1975, 'testB': 1810} #### RefCOCO
  print('num_refs:', num_refs[split])
  # all detections are collected into:
  #  all_boxes[sent][cls] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[] for _ in range(81)]
  #all_boxes = [[[] for _ in range(81)]
  #       for _ in range(num_refs[split])]
  #  all_rles[sent][cls] = [rle] array of N rles
  #all_rles = [[[] for _ in range(81)]
  #       for _ in range(num_refs[split])]
  
  cum_I, cum_U = 0, 0
  eval_seg_iou_list = [.5, .6, .7, .8, .9]
  seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
  seg_total = 0
  
  while True:

    #data = loader.getTestBatch(split, opt)
    #det_ids = data['det_ids']
    #sent_ids = data['sent_ids']
    #Feats = data['Feats'] 
    #labels = data['labels']
    data = loader.getTestBatch(split)
    
    image = data['data']
    im_info = data['im_info']
    gt_boxes = data['gt_boxes'] # scaled
    gt_masks = data['gt_masks']
    labels = data['labels']
    file_name = data['file_name']
    bounds = data['bounds']
    
    blobs = {}
    blobs['data'] = image
    blobs['im_info'] = im_info
    blobs['file_name'] = file_name
    blobs['bounds'] = bounds

    
    ##print('------------------------------------')
    #for i, sent_id in enumerate(sent_ids):
    for i in range(labels.shape[0]):
      blobs['gt_boxes'] = gt_boxes[i:i+1, :]
      blobs['gt_masks'] = gt_masks[i:i+1, :, :]
      label = labels[i:i+1, :]
      max_len = (label != 0).sum().data[0]
      blobs['labels'] = label[:, :max_len] # (1, max_len)
      blobs['sent_id'] = i
      
      
      scores, boxes, net_conv, im_scale = im_detect(model, blobs) # (n, 81), (n, 81*4), (n, 1024, H, W), float
      #print('--------')
      #print('scores:', scores.shape) # (266, 81) (300, 81)
      #print('boxes:', boxes.shape) # (266, 324) (300, 324) not scaled
      #print('net_conv:', net_conv.shape) # (1L, 1024L, 37L, 63L) (1L, 1024L, 37L, 63L)
      #print('im_scale:', im_scale) # 2.0 2.0
      
      # skip j = 0, because it's the background class
      for j in range(1, 81):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
          .astype(np.float32, copy=False)
        keep = nms(torch.from_numpy(cls_dets), cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
        cls_dets = cls_dets[keep, :]
        all_boxes[j] = cls_dets

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][:, -1]
                      for j in range(1, 81)])
        ##print('--------')
        if len(image_scores) > max_per_image:
          image_thresh = np.sort(image_scores)[-max_per_image]
          image_highest = np.max(image_scores)
          
          for j in range(1, 81):
            keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
            all_boxes[j] = all_boxes[j][keep, :] # choose largest 100 boxes for image i
            
            for k in range(len(all_boxes[j])):
              if all_boxes[j][k, -1] >= image_highest:
                pred_box = all_boxes[j][k][:4]
                pred_class = j

      gt_box = blobs['gt_boxes'][0, :4] / im_scale
      
      
      iou = computeIoU_box(pred_box, gt_box) # both original size
      ##print('IoU:', iou * 100)
      
      if iou >= 0.5:
        acc += 1
      loss_evals += 1


      # run mask branch on all_boxes[i][:]
      accumulated_boxes = np.array([pred_box])
      accumulated_labels = np.array([pred_class])
      
      #accumulated_boxes  = []
      #accumulated_labels = []
      #for j in range(1, 81):
      #  if all_boxes[i][j].shape[0] > 0:
      #    accumulated_boxes += [all_boxes[i][j][:, :4]]
      #    accumulated_labels += [j]*all_boxes[i][j].shape[0]
      #accumulated_boxes = np.vstack(accumulated_boxes)   # accumulate max_per_image boxes [xyxy] (100, 4)
      #accumulated_labels = np.array(accumulated_labels, dtype=np.uint8) # n category labels
      
      mask_prob = model._predict_masks_from_boxes_and_labels(net_conv, 
                              accumulated_boxes * im_scale,  # scaled boxes [xyxy]
                              accumulated_labels) # (n, 14, 14)
      mask_prob = mask_prob.data.cpu().numpy() # convert to numpy
      #print('accumulated_boxes:', accumulated_boxes, accumulated_boxes.shape) # (100, 4)
      #print('accumulated_labels:', accumulated_labels, accumulated_labels.shape) # (100,)
      #print('mask_prob:', np.min(mask_prob), np.max(mask_prob), mask_prob.shape) # 0~1 float (100, 14, 14)
      #print('size:', int(round(blobs['im_info'][0][0]/im_scale)), int(round(blobs['im_info'][0][1]/im_scale)))
      pred_mask = recover_masks(mask_prob, accumulated_boxes, int(round(blobs['im_info'][0][0]/im_scale)), int(round(blobs['im_info'][0][1]/im_scale))) # (n, ih, iw) uint8 [0,1]
      #print('pred_mask 0:', np.unique(pred_mask), pred_mask.shape) # 0~255 int (100, 294, 500)
      
      pred_mask = np.squeeze((pred_mask > 122.).astype(np.uint8), axis=0)  # (n, ih, iw) uint8 [0,1] original size
      #print('pred_mask 1:', np.unique(pred_mask), pred_mask.shape)
      
      # add to all_rles
      #rles = [COCOmask.encode(np.asfortranarray(m)) for m in pred_mask]
      #ri = 0
      #for j in range(1, 81):
      #  ri_next = ri+all_boxes[i][j].shape[0]
      #  all_rles[i][j] = rles[ri:ri_next]
      #  assert len(all_rles[i][j]) == all_boxes[i][j].shape[0]
      #  ri = ri_next
      
      gt_mask = imresize(np.squeeze(blobs['gt_masks'], axis=0), size=pred_mask.shape, interp='nearest')
      
      # compute iou
      I, U = computeIoU_seg(pred_mask, gt_mask)
      cum_I += I
      cum_U += U
      for n_eval_iou in range(len(eval_seg_iou_list)):
        eval_seg_iou = eval_seg_iou_list[n_eval_iou]
        seg_correct[n_eval_iou] += (I*1.0/U >= eval_seg_iou)
      seg_total += 1
      

      # add info
      #entry = {}
      
      #entry['file_name'] = file_name
      #entry['sent'] = loader.decode_labels(blobs['labels'].data.cpu().numpy())[0] # gd-truth sent
      #entry['gt_box'] = gt_box
      #entry['pred_box'] = pred_box
      #predictions.append(entry)
      ##print(i, ':', entry['sent'])
      num_sent += 1

      # if used up
      if num_sents > 0 and loss_evals >= num_sents:
        finish_flag = True
        break
      """
      # add back mean
      image_vis = image + cfg.PIXEL_MEANS
      image_vis = imresize(image_vis[0], np.round(im_info[0][:2] / im_info[0][2])) # assume we only have 1 image
      # BGR to RGB (opencv uses BGR)
      image_vis = image_vis[np.newaxis, :,:,::-1].copy(order='C')
      
      pred_box_vis = np.append(pred_box * im_scale, pred_class)
      pred_box_vis = np.expand_dims(pred_box_vis, axis=0)
      
      #print('image_vis:', image_vis.shape)
      #print('gt_boxes:', blobs['gt_boxes'], blobs['gt_boxes'].shape)
      #print('pred_box:', pred_box_vis, pred_box_vis.shape)
      #print('im_info:', im_info, im_info.shape)
      
      box_gt = draw_bounding_boxes(image_vis.copy(), blobs['gt_boxes'], im_info)
      box_pred = draw_bounding_boxes(image_vis.copy(), pred_box_vis, im_info)
      
      image_box_gt = Image.fromarray(box_gt[0, :])
      image_box_pred = Image.fromarray(box_pred[0, :])
      
      box_dir = 'result_box'
      if not os.path.exists(box_dir):
        os.makedirs(box_dir)
      image_box_gt.save('{}/{}_{}_box_gt.png'.format(box_dir, file_name[:-4], i))
      image_box_pred.save('{}/{}_{}_box_pred.png'.format(box_dir, file_name[:-4], i))

      
      # gt seg
      seg_gt = Image.fromarray(gt_mask*255).convert('L')
      seg_gt_c = np.array(seg_gt.filter(ImageFilter.CONTOUR))
      seg_gt_c = np.expand_dims(seg_gt_c, axis=2)
      seg_gt_c = np.concatenate((seg_gt_c, seg_gt_c, seg_gt_c), axis=2)
      
      seg_gt = np.array(seg_gt)
      seg_gt = np.expand_dims(seg_gt, axis=2)
      
      image_seg_gt = np.squeeze(box_gt, axis=0) + 0.5 * np.concatenate((seg_gt, seg_gt*0, seg_gt*0), axis=2)
      image_seg_gt[seg_gt_c==0] = 255
      image_seg_gt[image_seg_gt>255] = 255
      image_seg_gt = Image.fromarray(image_seg_gt.astype('uint8'))
      
      # pred seg
      seg_pred = Image.fromarray(pred_mask*255).convert('L')
      seg_pred_c = np.array(seg_pred.filter(ImageFilter.CONTOUR))
      seg_pred_c = np.expand_dims(seg_pred_c, axis=2)
      seg_pred_c = np.concatenate((seg_pred_c, seg_pred_c, seg_pred_c), axis=2)
      
      seg_pred = np.array(seg_pred)
      seg_pred = np.expand_dims(seg_pred, axis=2)
      
      image_seg_pred = np.squeeze(box_pred, axis=0) + 0.5 * np.concatenate((seg_pred, seg_pred*0, seg_pred*0), axis=2)
      image_seg_pred[seg_pred_c==0] = 255
      image_seg_pred[image_seg_pred>255] = 255
      image_seg_pred = Image.fromarray(image_seg_pred.astype('uint8'))
      
      # save seg
      seg_dir = 'result_box_seg'
      if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
      image_seg_gt.save('{}/{:0>5d}_{}_{}_seg_gt.png'.format(seg_dir, loader.iterators[split], file_name[:-4], i))
      image_seg_pred.save('{}/{}_{}_seg_pred.png'.format(seg_dir, file_name[:-4], i))
      """

      torch.cuda.empty_cache()


    # print
    ix0 = bounds['it_pos_now']
    ix1 = bounds['it_max']
    if verbose:
      print('evaluating [%s] ... image[%d/%d]\'s sents, det acc=%.2f%%, seg acc=%.2f%%, seg IoU=%.2f%%' % \
            (split, ix0, ix1, acc*100.0/loss_evals, seg_correct[0]*100.0/seg_total, cum_I*100.0/cum_U))

    # if we wrapped around the split
    if finish_flag or bounds['wrapped']:
      break
  return acc/loss_evals, eval_seg_iou_list, seg_correct, seg_total, cum_I, cum_U, num_sent
