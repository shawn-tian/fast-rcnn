# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.pascal_voc
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class ViDetectionData(datasets.imdb):
    def __init__(self, image_info):
        datasets.imdb.__init__(self, 'none')
        self._classes = image_info.get('label_set', None)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        #self._image_ext = '.jpg'
        self._image_folder = image_info['folder']
        self._image_index = [im['name'] for im in image_info['images']]
        self._image_info = image_info
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        #self.config = {'cleanup'  : True,
                       #'use_salt' : True,
                       #'top_k'    : 2000}

        #assert os.path.exists(self._devkit_path), \
                #'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        #assert os.path.exists(self._data_path), \
                #'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._image_folder, index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _parse_gt_info(self):
        all_box = self._image_info['images']
        num_objs = len(all_box)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        for ix, box_info in enumerate(all_box):
            x1, y1, x2, y2 = [int(float(s)) for s \
                    in box_info['x1y1x2y2'].split()]
            boxes[ix, :] = [x1, y1, x2, y2]
            if is_has_label:
                label_name = box_info.get('label', None)
                cls = self._class_to_ind[box_info['label']]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def selective_search_roidb(self):
        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        ss_roidb = [self._load_pascal_annotation(curr_image, False) \
                for curr_image in self._image_info['images']]
        box_list = [roidb['boxes'] for roidb in ss_roidb]
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def gt_roidb(self):
        gt_roidb = [self._load_pascal_annotation(curr_image, True) \
                for curr_image in self._image_info['images']]
        return gt_roidb
    
    def _select_boxes(self, all_box, is_gt):
        result = []
        for box in all_box:
            label = box.get('label', None)
            if is_gt:
                if label != None and self._class_to_ind[label] != 0:
                    result.append(box)
            else:
                if label == None or self._class_to_ind[label] == 0:
                    result.append(box)
        return result 

    def _load_pascal_annotation(self, curr_image, is_gt):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        all_box = curr_image['boxes']
        all_box = self._select_boxes(all_box, is_gt)
        num_objs = len(all_box)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        for ix, box_info in enumerate(all_box):
            x1, y1, x2, y2 = [int(float(s)) for s \
                    in box_info['x1y1x2y2'].split()]
            boxes[ix, :] = [x1, y1, x2, y2]
            label_name = box_info.get('label', None)
            if is_gt:
                cls = self._class_to_ind[label_name]
                assert cls > 0
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
