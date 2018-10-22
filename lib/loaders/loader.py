"""
data_json has
0. refs        : list of {ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds} (reference: box with 2~4 sentence)
{'box': [103.93, 299.99, 134.22, 177.42], 'att_wds': ['blue', 'lady', 'shirt'], 'sent_ids': [0, 1, 2], 'ann_id': 1719310, 'ref_id': 0, 'image_id': 581857, 'split': 'train', 'category_id': 1}

{'box': [452.46, 231.71, 175.67, 143.12], 'att_wds': ['black'], 'sent_ids': [117732, 117733, 117734], 'ann_id': 116865, 'ref_id': 41412, 'image_id': 98304, 'split': 'train', 'category_id': 63}

{'box': [334.49, 288.62, 130.02, 130.82], 'att_wds': [], 'sent_ids': [117735, 117736, 117737], 'ann_id': 108703, 'ref_id': 41413, 'image_id': 98304, 'split': 'train', 'category_id': 62}

{'box': [455.85, 231.54, 174.78, 145.55], 'att_wds': [], 'sent_ids': [117738, 117739, 117740], 'ann_id': 2222700, 'ref_id': 41414, 'image_id': 98304, 'split': 'train', 'category_id': 62}

{'box': [71.88, 213.93, 181.97, 199.26], 'att_wds': ['sofa'], 'sent_ids': [117741, 117742, 117743], 'ann_id': 99893, 'ref_id': 41415, 'image_id': 98304, 'split': 'train', 'category_id': 63}

1. images      : list of {image_id, ref_ids, ann_ids, file_name, width, height, h5_id}
{'h5_id': 0,
 'width': 640,
 'file_name': 'COCO_train2014_000000098304.jpg',
 'ref_ids': [41412, 41413, 41414, 41415],
 'image_id': 98304,
 'ann_ids': [3007, 99893, 108703, 115415, 116865, 120114, 1490146, 1511501, 1513367, 2099764, 2222700, 2222956],
 'height': 424}

{'h5_id': 1,
 'width': 640,
 'file_name': 'COCO_train2014_000000131074.jpg',
 'ref_ids': [38648, 38649],
 'image_id': 131074,
 'ann_ids': [318235, 319598, 1174042, 1630619, 1957252],
 'height': 428}

2. anns        : list of {ann_id, category_id, image_id, box, h5_id}  (RoI, not all have reference sentence)
{'box': [263.87, 216.88, 21.13, 15.17], 'image_id': 98304, 'h5_id': 0, 'category_id': 18, 'ann_id': 3007}
{'box': [71.88, 213.93, 181.97, 199.26], 'image_id': 98304, 'h5_id': 1, 'category_id': 63, 'ann_id': 99893}

3. sentences   : list of {sent_id, tokens, h5_id}
{'tokens': ['front', 'black', 'chair'], 'h5_id': 117732, 'sent_id': 117732}
{'tokens': ['right', 'couch', 'closest', '2', 'us'], 'h5_id': 117733, 'sent_id': 117733}
{'tokens': ['right', 'seat'], 'h5_id': 117734, 'sent_id': 117734}

{'tokens': ['center', 'chair', 'facing', 'away'], 'h5_id': 117735, 'sent_id': 117735}
{'tokens': ['middle', 'chair'], 'h5_id': 117736, 'sent_id': 117736}
{'tokens': ['tan', 'chair', 'center'], 'h5_id': 117737, 'sent_id': 117737}

4: word_to_ix  : word->ix
{'hats': 1465, 'yellow': 1, 'four': 2, 'asian': 937, 'hanging': 3, 'hate': 1466, 'whose': 938, 'feeding': 1467, 'pointing': 705, 'swan': 939, 'bike': 479, ...}

5: cat_to_ix   : cat->ix
{'toilet': 70, 'teddy bear': 88, 'sports ball': 37, 'bicycle': 2, 'kite': 38, 'stop sign': 13, 'tennis racket': 43, 'dog': 18, 'snowboard': 36, 'carrot': 57, ...}

6: label_length: L
Note, box in [xywh] format
data_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
self.data_h5['labels'][0,:] = [ 556  685  244  556 1478 1837    0    0    0    0]
self.data_h5['labels'][1,:] = [ 685  244 1322  260  364    0    0    0    0    0]
self.data_h5['labels'][2,:] = [1478 1837    0    0    0    0    0    0    0    0]
self.data_h5['labels'][3,:] = [1737 1028  882 1837 1203  830 1706  109    0    0]
self.data_h5['labels'][4,:] = [1737  882  109    0    0    0    0    0    0    0]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import h5py
import json
import random

class Loader(object):

    def __init__(self, data_json, data_h5=None):
        # load the json file which contains info about the dataset
        print('Loader loading data.json:', data_json)
        self.info = json.load(open(data_json))
        self.word_to_ix = self.info['word_to_ix']
        self.ix_to_word = {ix: wd for wd, ix in self.word_to_ix.items()}
        print('vocab size is', self.vocab_size) # 1999
        self.cat_to_ix = self.info['cat_to_ix']
        self.ix_to_cat = {ix: cat for cat, ix in self.cat_to_ix.items()}
        print('object cateogry size is', len(self.ix_to_cat)) # 80
        self.images = self.info['images']
        self.anns = self.info['anns']
        self.refs = self.info['refs']
        self.sentences = self.info['sentences']
        print('we have %s images.' % len(self.images)) # 19994
        print('we have %s anns.' % len(self.anns)) # 196771
        print('we have %s refs.' % len(self.refs)) # 50000
        print('we have %s sentences.' % len(self.sentences)) # 142210
        print('label_length is', self.label_length) # 10

        # construct mapping
        self.Refs = {ref['ref_id']: ref for ref in self.refs}
        self.Images = {image['image_id']: image for image in self.images}
        self.Anns = {ann['ann_id']: ann for ann in self.anns}
        self.Sentences = {sent['sent_id']: sent for sent in self.sentences}
        self.annToRef = {ref['ann_id']: ref for ref in self.refs}
        self.sentToRef = {sent_id: ref for ref in self.refs for sent_id in ref['sent_ids']}

        # read data_h5 if exists
        self.data_h5 = None
        if data_h5 is not None:
            print('Loader loading data.h5:', data_h5)
            self.data_h5 = h5py.File(data_h5, 'r')
            assert self.data_h5['labels'].shape[0] == len(self.sentences), 'label.shape[0] not match sentences' # 142210
            assert self.data_h5['labels'].shape[1] == self.label_length, 'label.shape[1] not match label_length' # 10

    @property
    def vocab_size(self):
        return len(self.word_to_ix)

    @property
    def label_length(self):
        return self.info['label_length']

    def encode_labels(self, sent_str_list):
        """Input:
        sent_str_list: list of n sents in string format
        return int32 (n, label_length) zeros padded in end
        """
        num_sents = len(sent_str_list)
        L = np.zeros((num_sents, self.label_length), dtype=np.int32)
        for i, sent_str in enumerate(sent_str_list):
            tokens = sent_str.split()
            for j, w in enumerate(tokens):
              if j < self.label_length:
                  L[i, j] = self.word_to_ix[w] if w in self.word_to_ix else self.word_to_ix['<UNK>']
        return L

    def decode_labels(self, labels):
        """
        labels: int32 (n, label_length) zeros padded in end
        return: list of sents in string format
        """
        decoded_sent_strs = []
        num_sents = labels.shape[0]
        for i in range(num_sents):
            label = labels[i].tolist()
            sent_str = ' '.join([self.ix_to_word[int(i)] for i in label if i != 0])
            decoded_sent_strs.append(sent_str)
        return decoded_sent_strs

    def fetch_label(self, ref_id, num_sents):
        """
        return: int32 (num_sents, label_length) and picked_sent_ids
        """
        ref = self.Refs[ref_id]
        sent_ids = list(ref['sent_ids'])  # copy in case the raw list is changed
        seq = []
        if len(sent_ids) < num_sents:
            append_sent_ids = [random.choice(sent_ids) for _ in range(num_sents - len(sent_ids))]
            sent_ids += append_sent_ids
        else:
            sent_ids = sent_ids[:num_sents]
        assert len(sent_ids) == num_sents
        # fetch label
        for sent_id in sent_ids:
            sent_h5_id = self.Sentences[sent_id]['h5_id']
            seq += [self.data_h5['labels'][sent_h5_id, :]]
        seq = np.vstack(seq)
        return seq, sent_ids

    def fetch_seq(self, sent_id):
        # return int32 (label_length, )
        sent_h5_id = self.Sentences[sent_id]['h5_id']
        seq = self.data_h5['labels'][sent_h5_id, :]
        return seq
