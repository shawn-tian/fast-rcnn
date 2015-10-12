#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import time, os, sys
import cv2
import numpy as np
from demo import vis_detections
import matplotlib.pyplot as plt
from google.protobuf import text_format
from pprint import pprint 
import yaml

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--label_set', dest='label_set',
                        help='label_set selected, seperated by comma.e.g. --label_set bag,shoe', 
                        default=None)
    parser.add_argument('--output_yaml', dest='output_yaml',
                        help='output_yaml_file', 
                        default=None)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    
    label_set = args.label_set
    print label_set
    if label_set != None:
        label_set = label_set.split(',')
    imdb = get_imdb(args.imdb_name, 
            dict(label_set = label_set, num_selected = 2))
    imdb.competition_mode(args.comp_mode)

    proto_file = args.prototxt
    with open(proto_file, 'r') as fp:
        all_content = fp.read()
    net_proto = caffe.proto.caffe_pb2.NetParameter()
    text_format.Merge(all_content, net_proto)
    all_finded = [layer for layer in net_proto.layer \
            if layer.name == 'cls_score']
    assert len(all_finded) == 1
    all_finded[0].inner_product_param.num_output = len(imdb.classes)
    all_finded = [layer for layer in net_proto.layer \
            if layer.name == 'bbox_pred']
    assert len(all_finded) == 1
    all_finded[0].inner_product_param.num_output = 4 * len(imdb.classes)
    with open(proto_file + '.out', 'w') as fp:
        fp.write(str(net_proto))

    net = caffe.Net(proto_file + '.out', args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    dets = test_net(net, imdb)
    imdb.evaluate_detections(dets)
    result = []
    if 1:
        classes = imdb.classes
        idx_image = 0
        for j, index in enumerate(imdb.image_index):
            curr_boxes = []
            for i, cls in enumerate(classes):
                if i == 0:
                    continue
                box_info = dets[i][j]
                if type(box_info) is list and len(box_info) == 0:
                    continue
                assert type(box_info) == np.ndarray, (box_info, type(box_info))
                im = cv2.imread(imdb.image_path_at(j))
                vis_detections(im, str(cls), box_info, thresh= 0.5)
                plt.show()
                ind = box_info[:, -1] > 0.5
                selected = box_info[ind, :]
                for k in range(selected.shape[0]):
                    s = selected[k, :]
                    curr_boxes.append({'x1y1x2y2': '{} {} {} {}'.format(int(s[0]), int(s[1]), int(s[2]), int(s[3])), \
                            'label': cls, \
                            'confidence': float(s[4])})
            curr_result = {'name': imdb.image_path_at(j), \
                    'boxes': curr_boxes}
            result.append(curr_result)
        result = dict(images = result)
        pprint(result)
        output_yaml = args.output_yaml
        if output_yaml == None:
            output_yaml = 'output_yaml'
        with open(output_yaml, 'w') as fp:
            yaml.dump(result, fp, Dumper = yaml.CDumper, 
                    default_flow_style = False)
