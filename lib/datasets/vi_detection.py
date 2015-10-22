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
from utils.cython_bbox import bbox_overlaps
from pprint import pprint

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
        self.all_matched_threshold = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
            if type(box_info['x1y1x2y2']) == str:
                x1, y1, x2, y2 = [int(float(s)) for s \
                        in box_info['x1y1x2y2'].split()]
            else:
                x1, y1, x2, y2 = box_info['x1y1x2y2']
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
            if type(box_info['x1y1x2y2']) == str:
                x1, y1, x2, y2 = [int(float(s)) for s \
                        in box_info['x1y1x2y2'].split()]
            else:
                x1, y1, x2, y2 = box_info['x1y1x2y2']
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

    #def competition_mode(self, on):
        #if on:
            #self.config['use_salt'] = False
            #self.config['cleanup'] = False
        #else:
            #self.config['use_salt'] = True
            #self.config['cleanup'] = True

    def _evaluate_one_class(self, retrieved_boxes, gt_boxes, output_dir = None):
        num_images = len(retrieved_boxes)
        all_is_matched = [] * num_images
        all_matched = np.zeros((len(self.all_matched_threshold), 0), dtype = np.bool)
        all_confidence = np.zeros((0,), dtype = np.float)
        total_gt = 0
        for i in range(num_images):
            curr_retrieved = retrieved_boxes[i]
            curr_gt = gt_boxes[i]
            total_gt = total_gt + curr_gt.shape[0]
            if len(curr_gt) == 0 or len(curr_retrieved) == 0:
                curr_matched = np.zeros((len(self.all_matched_threshold), len(curr_retrieved)), \
                        dtype = np.bool)
                curr_confidence = np.zeros((len(curr_retrieved), ))
            else:
                curr_matched = self._matched_information(curr_retrieved[:, :4], curr_gt)
                curr_confidence = curr_retrieved[:, 4]
            all_confidence = np.hstack((all_confidence, curr_confidence))
            all_matched = np.hstack((all_matched, curr_matched))
        if total_gt == 0:
            all_ap = np.zeros((len(self.all_matched_threshold)))
        else:
            idx = all_confidence.argsort()[::-1]
            indicator = all_matched[:, idx]
            x = np.cumsum(indicator, axis = 1)
            prec = x / np.array(range(1, x.shape[1] + 1), dtype = np.float)
            rec = x / total_gt 
            all_ap = self._get_all_ap(prec, rec)
        return all_ap

    def _get_all_ap(self, all_prec, all_rec):
        all_ap = np.zeros((len(self.all_matched_threshold), ), dtype =
                np.float)
        for i in range(len(self.all_matched_threshold)):
            prec, rec = all_prec[i], all_rec[i]
            ap=0;
            for j in range(11):
                t = j * 0.1
                ind = rec>=t
                if any(ind):
                    p=max(prec[ind])
                else:
                    p = 0
                ap=ap + p/11.0;
            all_ap[i] = ap
        return all_ap

    def _matched_information(self, curr_retrieved, curr_gt):
        is_matched = np.zeros((len(self.all_matched_threshold), 
            len(curr_retrieved)), dtype = np.bool)
        if len(curr_retrieved) == 0 or len(curr_gt) == 0:
            return is_matched 
        else:
            gt_overlaps = bbox_overlaps(curr_retrieved.astype(np.float),
                                        curr_gt.astype(np.float))
            matched_idx = gt_overlaps.argmax(axis = 1)
            for k in range(len(self.all_matched_threshold)):
                matched_threshold = self.all_matched_threshold[k]
                gt_used = np.zeros(len(curr_gt), dtype = np.bool)
                for i, j in enumerate(matched_idx):
                    if gt_overlaps[i, j] >= matched_threshold and \
                            gt_used[j] == False:
                        gt_used[j] = True
                        is_matched[k, i] = True
        return is_matched

    def evaluate_detections(self, all_boxes, output_dir=None):
        assert len(all_boxes) == self.num_classes
        for boxes in all_boxes:
            assert(len(boxes) == self.num_images)
        all_ap = np.zeros((len(self.all_matched_threshold), (len(all_boxes) -
            1)))
        for i, boxes in enumerate(all_boxes):
            # ignore the background
            if i == 0: 
                continue
            curr_roidb = self._retrieve_gt(i)
            curr_ap = self._evaluate_one_class(boxes, curr_roidb, output_dir)
            all_ap[:, i - 1] = curr_ap
        res = all_ap.mean(axis = 1)
        pprint(zip(self.all_matched_threshold, res))
        return zip(self.all_matched_threshold, res)

    def _retrieve_gt(self, idx_class):
        assert len(self.roidb) == self.num_images
        result = [[]] * self.num_images
        for i in range(self.num_images):
            curr_rois = self.roidb[i]
            all_idx = curr_rois['gt_classes'] == idx_class
            assert curr_rois['flipped'] == False 
            if len(all_idx) > 0:
                result[i] = curr_rois['boxes'][all_idx, :]
        return result
            

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
