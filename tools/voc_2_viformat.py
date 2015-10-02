import os
import yaml
import cPickle as pickle

fn = '/home/jianfeng/code/fast-rcnn/data/cache/voc_2007_trainval_gt_roidb.pkl'
with open(fn, 'r') as fp:
    bb_info = pickle.load(fp)

fn = '/home/jianfeng/code/fast-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt'
with open(fn, 'r') as fp:
    all_line = fp.readlines()
image_folder = '/home/jianfeng/code/fast-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages'
all_image_name = [line.strip() + '.jpg' for line in all_line]
print len(all_image_name)
print len(bb_info)

all_out_image = []
for i in range(len(bb_info)):
    image_name = all_image_name[i]
    bbs = bb_info[i]['boxes']
    labels = bb_info[i]['gt_classes']
    assert bbs.shape[0] == len(labels)
    out = {'name': image_name}
    curr_all_box = []
    for j in range(bbs.shape[0]):
        bb = bbs[j]
        label = labels[j]
        curr_all_box.append(dict(x1y1x2y2 = ' '.join(str(b) for b in bb), \
                label = int(label)))
    all_out_image.append(dict(name = image_name, \
            boxes = curr_all_box))
label_set = range(21) 
output = 'yy.yaml'
with open(output, 'w') as fp:
    yaml.dump(dict(folder = image_folder, \
            label_set = label_set, \
            images = all_out_image), \
            fp, default_flow_style = False)


