#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import pdb
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import imghdr

DEBUG = False

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__',
            'shoe', 'bag', 'dress', 'top', 'pant', 'skirt')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

out_path = '/home/shangxuan/visenzeWork/data-platform/tasks/faster_rcnn_test/'

def vis_detections(im, class_name, dets, im_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    head, tail = os.path.split(im_name)
    out_folder = os.path.basename(head)
    out_img_name = os.path.join(out_path, out_folder, tail)
    plt.savefig(out_img_name)
    plt.close(fig)

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    if not cfg.TEST.HAS_RPN:
        # Load pre-computed Edgebox object proposals
        fpath, tail = os.path.split(image_name)
        fname = os.path.splitext(tail)[0]
        box_file = os.path.join(fpath, fname + '.mat')
        obj_proposals = sio.loadmat(box_file)['boxes']

    # Load the demo image
    # im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name)
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    if not cfg.TEST.HAS_RPN:
        scores, boxes = im_detect(net, im, obj_proposals)
    else:
        scores, boxes = im_detect(net, im)

    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3

    res_all = []
    score_max = 0
    cls_max = []
    dets_max = []

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        if len(keep) < 1:
            continue
        dets = dets[[0]] # take the highest in each class
        res_box = {}
        res_box['class'] = cls
        res_box['box'] = dets[0, 0:4]
        res_box['score'] = dets[0, 4]
        res_all.append(res_box)

        if res_box['score'] > score_max:
            cls_max = cls
            dets_max = dets
            score_max = res_box['score']

    if score_max > CONF_THRESH:
        vis_detections(im, cls_max, dets_max, image_name, thresh=CONF_THRESH)
    
    # save the detected boxes and classes
    head, tail = os.path.split(image_name)
    fname = os.path.splitext(tail)[0]
    out_folder = os.path.basename(head)
    # txt_file = os.path.join(out_path, out_folder, fname+'.txt')
    # fp = open(txt_file, "w")
    # for box in res_all:
    #     rect = box['box']
    #     fp.write("%s %s %.6f %d %d %d %d" % (fname, box['class'], box['score'], rect[0], rect[1], rect[2], rect[3]) )
    # fp.close()

    mat_file = os.path.join(out_path, out_folder, fname+'.mat')
    sio.savemat(mat_file, mdict={'det_res': res_all})

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True
      # Use RPN for proposals

    args = parse_args()

    # prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
    #                         'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    # caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'faster_rcnn_models',
    #                           NETS[args.demo_net][1])
    
    # load faster rcnn model or fast rcnn model
    if cfg.TEST.HAS_RPN:
        prototxt = os.path.join('/home/shangxuan/visenzeWork/data-platform/tasks/faster_rcnn_vgg/faster_rcnn_None',
                            'deploy.prototxt')
        caffemodel = os.path.join('/home/shangxuan/visenzeWork/data-platform/tasks/faster_rcnn_vgg/faster_rcnn_None',
                           'vgg_cnn_m_1024_faster_rcnn_iter_300000.caffemodel')
    else:
        prototxt = os.path.join('/home/shangxuan/visenzeWork/data-platform/tasks/frcnn/frcnn_None',
                            'deploy.prototxt')
        caffemodel = os.path.join('/home/shangxuan/visenzeWork/data-platform/tasks/frcnn/frcnn_None', 
                            'caffenet_fast_rcnn_iter_80000.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    # im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _= im_detect(net, im)

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']
    # im_names = ['obj_shoe_005.jpg', 'obj_shoe_006.jpg', 
    #             'obj_shoe_007.jpg', 'obj_shoe_008.jpg']
    # for im_name in im_names:

    if 0:
        pdb.set_trace()
        im_names = ['000542.jpg', 'bag.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
        for im_name in im_names:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo for data/demo/{}'.format(im_name)
            demo(net, im_name)

    input_path = '/mnt/distribute_env/usr/xf/rcnn_test/'
    test_class = 'dress'
    for (dirpath, dirnames, filenames) in os.walk(input_path+test_class):
        for im_names in filenames:
            if im_names[0] == '.':
                continue
            im_name = os.path.join(input_path, test_class, im_names)
            
            if imghdr.what(im_name) == None:
                continue
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Detection for {}'.format(im_name)
            demo(net, im_name)
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    plt.show()
