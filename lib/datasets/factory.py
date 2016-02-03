# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import sys
import time
import datasets.pascal_voc
import numpy as np
import os
import yaml
from datasets.vi_detection import ViDetectionData
import cPickle as pickle
from multiprocessing import Pool
import logging
import PIL
import pdb

logging.basicConfig(level = logging.INFO, \
        format = '[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

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

def get_yaml_files(folder, num_selected):
    sub_names = [];
    for dir_path, sub_dir, all_sub_file in os.walk(folder, followlinks=True):
        for sub_file in all_sub_file:
            if sub_file.endswith('.yml'): #or \
                    #sub_file.endswith('.yml'):
                full_name = os.path.join(dir_path, sub_file);
                x = full_name.replace(folder, '');
                if len(x) >= 1 and x[0] == '/':
                    x = x[1 : ];
                sub_names.append(x);
    if num_selected > 0:
        num_selected = min(num_selected, len(sub_names))
        return sub_names[: num_selected]
    else:
        return sub_names

def clamp_box(data_config):
    print 'Begin to clamp box...'
    all_image_info = data_config['images']
    folder = data_config['folder']
    to_remove_image = []
    for image_info in all_image_info:
        all_box = image_info['boxes']
        fn = os.path.join(folder, image_info['name'])
        try:
            width, height = PIL.Image.open(fn).size
        except:
            print 'not available: {}'.format(fn)
            to_remove_image.append(image_info)
            continue
        num_box_org = len(all_box)
        to_be_remove = []
        for box in all_box:
            bb_str = box['x1y1x2y2']
            x1, y1, x2, y2 = [int(float(s)) for s in bb_str.split()]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, width - 1)
            y2 = min(y2, height - 1)
            if x2 < x1 or y2 < y1:
                to_be_remove.append(box)
            else:
                bb_str = '{} {} {} {}'.format(x1, y1, x2, y2)
                box['x1y1x2y2'] = bb_str
        for box in to_be_remove:
            all_box.remove(box)
        if len(to_be_remove) > 0:
            logger.info('box removed: {}-->{}'.format(num_box_org,
                len(all_box)))
    for image_info in to_remove_image:
        all_image_info.remove(image_info)

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

def add_missing_labels(all_image_info):
    for image_info in all_image_info:
        all_box = image_info['boxes']
        for box_info in all_box:
            if not box_info.has_key('label') or \
                'background' in box_info['label']:
                box_info['label'] = '__background__'

def remove_unknown_box(data_config):
    label_set = data_config['label_set']
    all_image_info = data_config['images']
    for image_info in all_image_info:
        all_box = image_info['boxes']
        to_be_removed = []
        for box_info in all_box:
            if box_info['label'] not in label_set:
                to_be_removed.append(box_info)
        for box_info in to_be_removed:
            all_box.remove(box_info)

def check_format(info):
    assert type(info) == dict, info
    assert info.has_key('images'), info
    all_image_info = info['images']
    assert type(all_image_info) == list, info
    for image_info in all_image_info:
        assert type(image_info) == dict, info
        assert image_info.has_key('name'), info
        assert image_info.has_key('boxes'), info
        all_box_info = image_info['boxes']
        assert type(all_box_info) == list, info
        is_use = False
        for box_info in all_box_info:
            assert type(box_info) == dict, info
            #assert box_info.has_key('label'), \
                    #(info, box_info)
            #assert type(box_info['label']) == str, \
                    #(info, box_info)
            assert box_info.has_key('x1y1x2y2'), \
                    (info, box_info)
            assert type(box_info['x1y1x2y2']) == str, \
                    (info, box_info)
            

def parallel_instance_get_imdb(info):
    folder, yaml_file = info
    full_yaml_file = os.path.join(folder, yaml_file)
    with open(full_yaml_file, 'r') as fp:
        x = yaml.load(fp, Loader = yaml.CLoader)
        try:
            check_format(x)
        except:
            logger.info('Format is illegal:{}, and ignored'.format(full_yaml_file))
            return None
    #curr_folder = x['folder']
    curr_folder = os.path.join(
            os.path.dirname(os.path.dirname(full_yaml_file)), 
            'JPEGImages')
    for image_info in x['images']:
        image_info['name'] = os.path.join(curr_folder, image_info['name'])
    #logger.info('File {} has been loaded'.format(full_yaml_file))
    return x

def parallel_get_imdb_folder_yaml(folder, label_set = None, num_selected = 0):
    all_yaml_file = get_yaml_files(folder, num_selected)
    result = None
    logger.info("begin parallel loading yaml files")
    pool = Pool(64)
    jobs = pool.map_async(parallel_instance_get_imdb, \
            zip([folder] * len(all_yaml_file), all_yaml_file), chunksize=1)
    while not jobs.ready():
        logger.info('left: {}'.format(jobs._number_left))
        time.sleep(10)
    pool.close()
    pool.join()
    all_job_result = jobs.get()
    result = {'images': []}
    for job_result in all_job_result:
        if job_result != None:
            result['images'].extend(job_result['images'])
    result['folder'] = '/'
    logger.info("finish parallel loading yaml files")
    logger.info("begin to add background label if not exist")
    add_missing_labels(result['images'])
    logger.info("finish adding background label if not exist")
    # if label_set == None:
    #     result['label_set'] = infer_label_set(result['images'])
    # else:
    #     all_bg = [label for label in label_set if 'background' in label.lower()]
    #     for bg in all_bg:
    #         label_set.remove(bg)
    #     result['label_set'] = ['__background__'] + label_set
    #     remove_unknown_box(result)
    return result

def sequence_get_imdb_folder_yaml(folder, label_set, num_selected):
    all_yaml_file = get_yaml_files(folder, num_selected)
    result = None
    for idx, yaml_file in enumerate(all_yaml_file):
        if (idx % 100) == 0:
            print 'loading the {}/{} yaml'.format(idx, len(all_yaml_file))
        full_yaml_file = os.path.join(folder, yaml_file)
        with open(full_yaml_file, 'r') as fp:
            x = yaml.load(fp, Loader = yaml.CLoader)
            check_format(x)
        #curr_folder = x['folder']
        curr_folder = os.path.join(
                os.path.dirname(os.path.dirname(full_yaml_file)), 
                'JPEGImages')
        for image_info in x['images']:
            image_info['name'] = os.path.join(curr_folder, image_info['name'])
        x['folder'] = '/'
        if result == None:
            result = x
        else:
            assert result['folder'] == x['folder'], \
                    (result['folder'], x['folder'])
            result['images'].extend(x['images'])
    add_missing_labels(result['images'])
    if label_set == None:
        result['label_set'] = infer_label_set(result['images'])
    else:
        all_bg = [label for label in label_set if 'background' in label.lower()]
        for bg in all_bg:
            label_set.remove(bg)
        result['label_set'] = ['__background__'] + label_set
        remove_unknown_box(result)
    return result

def get_imdb_folder_yaml(folder, label_set = None, num_selected = 0):
    if True:
        return parallel_get_imdb_folder_yaml(folder, label_set, num_selected)
    else:
        return sequence_get_imdb_folder_yaml(folder, label_set, num_selected)

def get_imdb_yaml(name, label_set = None):
    print 'loading the data'
    with open(name, 'r') as fp:
        image_info = yaml.load(fp, Loader = yaml.CLoader)
    print 'finish loading'
    add_missing_labels(image_info['images'])
    if not image_info.has_key('label_set'):
        image_info['label_set'] = infer_label_set(image_info['images'])
    if label_set != None:
        all_bg = [label for label in label_set if 'background' in label.lower()]
        for bg in all_bg:
            label_set.remove(bg)
        image_info['label_set'] = ['__background__'] + label_set
        remove_unknown_box(image_info)
    from pprint import pprint
    pprint(image_info['label_set'])
    return image_info

def remove_images_no_label(all_info):
    logger.info("begin removing images with no labels")
    all_image_info = all_info['images']
    logger.info("There are {} images".format(len(all_image_info)))
    to_be_remove = []
    for image_info in all_image_info:
        all_box = image_info['boxes']
        is_use = False
        for box in all_box:
            if box['label'] != '__background__':
                is_use = True
                break
        if is_use == False:
            to_be_remove.append(image_info)
    logger.info("{} images will be removed".format(len(to_be_remove)))
    for image_info in to_be_remove:
        logger.info('no label founded and remove: {}'.format(image_info['name']))
        all_image_info.remove(image_info)
    logger.info('after remove {}'.format(len(all_image_info)))

def remove_unknown_box_and_map(all_info, cls_mapping = None):
    print 'Removing unkonwn box and mapping the detection classes ...'
    label_set = all_info['label_set']
    all_image_info = all_info['images']
    for image_info in all_image_info:
        all_box = image_info['boxes']
        to_be_removed = []
        if cls_mapping is None:
            for box_info in all_box:
                if box_info['label'] not in label_set:
                    to_be_removed.append(box_info)
        else:
            # map the label first
            for box_info in all_box:
                if box_info['label'] in label_set:
                    continue
                box_label_map = cls_mapping.get(box_info['label'], None)
                if box_label_map is None or box_label_map not in label_set:
                    to_be_removed.append(box_info)
                else:
                    box_info['label'] = box_label_map

        for box_info in to_be_removed:
            all_box.remove(box_info)

def get_imdb(name, param = {}):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):

        if os.path.exists(name):
            # loads/saves from/to a cache file to speed up future calls
            cache_dir = param['cache_dir']
            output_dir = param['output_dir']
            output_cache_name = os.path.basename(os.path.dirname(output_dir))
            cache_file = os.path.join(cache_dir, output_cache_name + '_imdb_train.pkl')
            if os.path.exists(cache_file): # load from cache
                with open(cache_file, 'rb') as fid:
                    image_info = pickle.load(fid)
                    logger.info('image info loaded from {}'.format(cache_file))
            else:
                if os.path.isdir(name):
                    image_info = get_imdb_folder_yaml(name, 
                            param.get('label_set', None), 
                            param.get('num_selected', 0))
                elif os.path.isfile(name):
                    image_info = get_imdb_yaml(name, param.get('label_set', None))
                else:
                    assert False, 'what is {}'.format(name)
                clamp_box(image_info)
                # save to cache file
                # with open(cache_file, 'wb') as fid:
                #     pickle.dump(image_info, fid, pickle.HIGHEST_PROTOCOL)
                #     logger.info('image info saved to {}'.format(cache_file))
            
            # remove images whose label is not in the label set
            # and map a set of small classes into a few larger categories
            label_set = param.get('label_set', None)
            if label_set == None:
                image_info['label_set'] = infer_label_set(image_info['images'])
            else:
                all_bg = [label for label in label_set if 'background' in label.lower()]
                for bg in all_bg:
                    label_set.remove(bg)
                image_info['label_set'] = ['__background__'] + label_set
                remove_unknown_box_and_map(image_info, param.get('cls_mapping', None))
            if param.get('is_remove_no_label', False):
                remove_images_no_label(image_info)
            return ViDetectionData(image_info)

        else:
            raise KeyError('Unknown dataset: {}'.format(name))
    else:
        return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

