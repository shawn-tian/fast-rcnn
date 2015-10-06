#!/usr/bin/env python
import os
import yaml
import cPickle as pickle

#fn = '/home/jianfeng/code/fast-rcnn/data/cache/voc_2007_trainval_gt_roidb.pkl'

_classes = ('__background__', # always index 0
                 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train', 'tvmonitor')
def gen_yaml(pkl_roi, image_list_file, output):
    with open(pkl_roi, 'r') as fp:
        bb_info = pickle.load(fp)
    with open(image_list_file, 'r') as fp:
        all_line = fp.readlines()
    image_folder = '/home/jianfeng/code/fast-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages'
    all_image_name = [line.strip() + '.jpg' for line in all_line]
    all_out_image = []
    for i in range(len(bb_info)):
    #for i in range(5):
        image_name = all_image_name[i]
        bbs = bb_info[i]['boxes']
        labels = bb_info[i]['gt_classes']
        assert bbs.shape[0] == len(labels)
        is_has_human = any([l == 15 for l in labels])
        if not is_has_human:
            continue
        curr_all_box = []
        for j in range(bbs.shape[0]):
            bb = bbs[j]
            label = labels[j]
            if label != 15 and label != 0:
                continue
            curr_all_box.append(dict(x1y1x2y2 = ' '.join(str(b) for b in bb), \
                    label = _classes[label]))
        all_out_image.append(dict(name = image_name, \
                boxes = curr_all_box))
        if len(all_out_image) > 1000:
            break
    print len(all_out_image)
    label_set = list(_classes)
    with open(output, 'w') as fp:
        yaml.dump(dict(folder = image_folder, \
                label_set = label_set, \
                images = all_out_image), \
                fp, default_flow_style = False, Dumper = yaml.CDumper)
    x = dict(folder = image_folder, \
                label_set = label_set, \
                images = all_out_image)
    with open('tmp.pkl', 'wb') as fp:
        pickle.dump(x, fp, pickle.HIGHEST_PROTOCOL)

#image_list_file = '/home/jianfeng/code/fast-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt'
#pkl_roi = '/home/jianfeng/code/fast-rcnn/data/cache/voc_2007_trainval_selective_search_roidb.pkl'
#output = 'data/human_train.yaml'
#gen_yaml(pkl_roi, image_list_file, output)

image_list_file = '/home/jianfeng/code/fast-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
pkl_roi = '/home/jianfeng/code/fast-rcnn/data/cache/voc_2007_test_selective_search_roidb.pkl'
output = 'data/human_ss.yaml'
gen_yaml(pkl_roi, image_list_file, output)

image_list_file= '/home/jianfeng/code/fast-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
pkl_roi = '/home/jianfeng/code/fast-rcnn/data/cache/voc_2007_trainval_gt_roidb.pkl'
output = 'data/human_gt.yaml'
gen_yaml(pkl_roi, image_list_file, output)

