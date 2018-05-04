import numpy as np
import os
import xml.etree.ElementTree as ET
import shutil
import argparse
import sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path('/home/wy/Documents/workspace/analysis/code/basic')
from cfgs.cfg_analysis_caffe import choose_dataset

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--db_str', dest='db_str',
                        help='database name in (SO, ST, SS, SF)',
                        default='SO', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

# args = parse_args()
# db_str = args.db_str
db_str = 'container/GeneralContainer/train/180206/letter'
# base_dir = os.path.join("/home/wy/disks/disk0/datasets", db_str)
base_dir = '/home/westwell/hj/hj_data/collect/all/char/'
img_dir = os.path.join(base_dir, 'data/JPEGImages')
ano_dir = os.path.join(base_dir, 'data/Annotations')
"""
db_dict = choose_dataset(db_str)
if db_dict:
    # print db_dict
    element_list = db_dict.ELEMENT_LIST
    element_dict = db_dict.ELEMENT_DICT
else:
    print 'error database string'
    sys.exit(0)
"""
db
cls_count = {}
for cls in element_list:
    cls_count[cls] = 0

def readAnoObject(filefpath_xml):
    tree = ET.parse(filefpath_xml)
    root = tree.getroot()
    return [obj.find('name').text for obj in root.findall('object')]


for imagename in os.listdir(img_dir):
    img_fpath = os.path.join(img_dir, imagename)
    ano_fpath = os.path.join(ano_dir, imagename[:-4] + '.xml')
    if os.path.exists(ano_fpath):
        ano_list = readAnoObject(ano_fpath)
        for cls in ano_list:
            cls_count[cls] = cls_count[cls] + 1

for k in element_list:
    if cls_count[k] !=0:
        print k, cls_count[k]
