# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.pascal_voc
import numpy as np
import os
import yaml
from datasets.vi_detection import ViDetectionData

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k))

def get_yaml_files(folder):
    sub_names = [];
    for dir_path, sub_dir, all_sub_file in os.walk(folder):
        for sub_file in all_sub_file:
            if sub_file.endswith('.yaml'):
                full_name = os.path.join(dir_path, sub_file);
                x = full_name.replace(folder, '');
                if len(x) >= 1 and x[0] == '/':
                    x = x[1 : ];
                sub_names.append(x);
    return sub_names

def infer_label_set(all_image_info):
    result = set()
    for image_info in all_image_info:
        for box in image_info['boxes']:
            l = box.get('label', None)
            if l != None:
                result.add(l)
    all_bg = []
    for res in result:
        if 'background' in res.lower():
            all_bg.append(res)
    for bg in all_bg:
        result.remove(bg)
    return ['__background__'] + list(result)

def get_imdb_folder_yaml(folder):
    all_yaml_file = get_yaml_files(folder)
    result = None
    for idx, yaml_file in enumerate(all_yaml_file):
        if (idx % 100) == 0:
            print 'loading the {}/{} yaml'.format(idx, len(all_yaml_file))
        with open(os.path.join(folder, yaml_file), 'r') as fp:
            x = yaml.load(fp, Loader = yaml.CLoader)
        curr_folder = x['folder']
        for image_info in x['images']:
            image_info['name'] = os.path.join(curr_folder, image_info['name'])
        x['folder'] = '/'
        if result == None:
            result = x
        else:
            assert result['folder'] == x['folder']
            result['images'].extend(x['images'])
    result['label_set'] = infer_label_set(result['images'])
    return ViDetectionData(result)

def get_imdb_yaml(name):
    print 'loading the data'
    with open(name, 'r') as fp:
        image_info = yaml.load(fp, Loader = yaml.CLoader)
    print 'finish loading'
    if not image_info.has_key('label_set'):
        image_info['label_set'] = infer_label_set(image_info['images'])
    from pprint import pprint
    pprint(image_info['label_set'])
    return ViDetectionData(image_info)

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        if os.path.exists(name):
            if os.path.isdir(name):
                return get_imdb_folder_yaml(name)
            elif os.path.isfile(name):
                return get_imdb_yaml(name)
            else:
                assert False, 'what is {}'.format(name)
        else:
            raise KeyError('Unknown dataset: {}'.format(name))
    else:
        return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

