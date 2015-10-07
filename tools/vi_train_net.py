#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
from google.protobuf import text_format


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--label_set', dest='label_set',
                        help='label_set selected, seperated by comma.e.g. --label_set bag,shoe', 
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
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    # set the number of class here
    label_set = args.label_set
    if label_set != None:
        label_set = label_set.split(',')
    imdb = get_imdb(args.imdb_name, dict(label_set = label_set, 
        is_remove_no_label = True))
    #print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    solver = args.solver
    if not cfg.TRAIN.BBOX_REG:
        import os
        solver = os.path.splitext(solver)
        solver = solver[0] + '_no_reg' + solver[1]

    # change the protocol file to incroporate the number of class
    num_class = len(imdb.classes)
    solver_param = caffe.proto.caffe_pb2.SolverParameter()
    with open(solver, 'r') as fp:
        all_content = fp.read()
    text_format.Merge(all_content, solver_param)
    net_file = solver_param.train_net
    net = caffe.proto.caffe_pb2.NetParameter()
    with open(net_file, 'r') as fp:
        all_content = fp.read()
    text_format.Merge(all_content, net)
    assert net.layer[0].type == 'Python'
    net.layer[0].python_param.param_str = "'num_classes': {}".format(num_class)
    all_finded = [layer for layer in net.layer if layer.name == 'cls_score']
    assert len(all_finded) == 1
    all_finded[0].inner_product_param.num_output = num_class
    all_finded = [layer for layer in net.layer if layer.name == 'bbox_pred']
    assert len(all_finded) == 1
    all_finded[0].inner_product_param.num_output = 4 * num_class
    with open(net_file + '.out', 'w') as fp:
        fp.write(str(net))
    solver_param.train_net = net_file + '.out'
    with open(solver + '.out', 'w') as fp:
        fp.write(str(solver_param))
    # train the model
    train_net(solver + '.out', roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)

