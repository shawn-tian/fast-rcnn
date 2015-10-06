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
def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

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
            if label == 0:
                curr_all_box.append(dict(x1y1x2y2 = ' '.join(str(b) for b in
                    bb)))
            else:
                curr_all_box.append(dict(x1y1x2y2 = ' '.join(str(b) for b in bb), \
                        label = _classes[label]))
        all_out_image.append(dict(name = image_name, \
                boxes = curr_all_box))
    print len(all_out_image)
    #label_set = list(_classes)
    for i in range(len(all_out_image)):
        bn = os.path.basename(all_out_image[i]['name'])
        bbn = os.path.splitext(bn)[0]
        fn = os.path.join(output, bbn + '.yaml')
        ensure_folder(os.path.dirname(fn))
        with open(fn, 'w') as fp:
            yaml.dump(dict(folder = image_folder, \
                    images = [all_out_image[i]]), \
                    fp, \
                    default_flow_style = False, Dumper = yaml.CDumper)
    print 'finished'

#image_list_file = '/home/jianfeng/code/fast-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt'
#pkl_roi = '/home/jianfeng/code/fast-rcnn/data/cache/voc_2007_trainval_selective_search_roidb.pkl'
#output = 'data/yaml_database/human_train'
#gen_yaml(pkl_roi, image_list_file, output)

#image_list_file = '/home/jianfeng/code/fast-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
#pkl_roi = '/home/jianfeng/code/fast-rcnn/data/cache/voc_2007_test_selective_search_roidb.pkl'
#output = 'data/yaml_database/human_test_ss'
#gen_yaml(pkl_roi, image_list_file, output)

image_list_file= '/home/jianfeng/code/fast-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
pkl_roi = '/home/jianfeng/code/fast-rcnn/data/cache/voc_2007_test_gt_roidb.pkl'
output = 'data/yaml_database/human_test_gt'
gen_yaml(pkl_roi, image_list_file, output)

