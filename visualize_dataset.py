import os,sys, shutil
import matplotlib.pyplot as plt
import Image
import numpy as np
import argparse
import threading

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path('/home/wy/Documents/workspace/analysis/code/basic')
from utils.da_utils import *
from cfgs.cfg_common import choose_dataset
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='test faster rcnn model')
    parser.add_argument('--db_str', dest='db_str',
                        help='model type one of [SO, ST, SS, SF]',
                        default='SF', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

base_dir = '/home/wy/disks/disk0/datasets'
args = parse_args()
print('Called with args:')
print(args)
db_str = args.db_str
img_dir = os.path.join(base_dir, db_str, 'data/JPEGImages')
ano_dir = os.path.join(base_dir, db_str, 'data/Annotations')
save_base_dir = '/home/wy/disks/disk0/datasets/vis'
save_dir = os.path.join(save_base_dir, db_str)
db_dict = choose_dataset(db_str)
if db_dict:
    print db_dict
    element_list = db_dict.ELEMENT_LIST
else:
    print 'error database string'
    sys.exit(0)

def check_dir(one_dir):
    if os.path.exists(one_dir):
        shutil.rmtree(one_dir)
    os.makedirs(one_dir)
    # for i in range(len(element_list)):
    #     os.makedirs(os.path.join(one_dir, element_list[i]))

check_dir(save_dir)
check_dir(os.path.join(save_dir, 'image'))
def vis_images(index,img_list, src_img_dir,src_ano_dir, save_dir):
    for imagename in img_list:
        # imagename = '44755.bmp'
        # if imagename != '14248.bmp':
        #     continue
        img_fpath = os.path.join(src_img_dir, imagename)
        view_img_fpath = os.path.join(save_dir,'image', imagename[:-4]+'.jpg')
        ano_fpath = os.path.join(src_ano_dir, imagename[:-4]+'.xml')
        if os.path.exists(img_fpath) and os.path.exists(ano_fpath):
            pass
        else:
            continue
        charlist, coords = readAnoXML(ano_fpath)
        # except_list = ['fj', 'hb', 'hl']
        # skip = True
        # for i in range(len(except_list)):
        #     if except_list[i] in charlist:
        #         skip = False
        # if skip:
        #     continue
        # if ''.join(charlist) != 'triu':
        #     continue
        im = cv2.imread(img_fpath)
        print index,' : Processing ==> ' ,img_fpath
        print im.shape

        for i in range(len(charlist)):
            # if charlist[i] not in ['zero','three','six','seven','eight','nine','a','b','c','e','h','k','m','p','q','s','t','u','w','x','y','z']:
            #     continue
            print index, ' : Processing ==> ', charlist[i]
            view_cls_fpath = os.path.join(save_dir, charlist[i], imagename[:-4]+'_'+str(i)+'.bmp')
            if not os.path.exists(os.path.join(save_dir, charlist[i])):
                os.makedirs(os.path.join(save_dir, charlist[i]))
            # print coords[i]
            tmp_im = im.copy()[coords[i][1]:coords[i][3], coords[i][0]:coords[i][2],:]
            cv2.imwrite(view_cls_fpath, tmp_im)
        saveImageWithCoordsClasses(im, coords, charlist, view_img_fpath)

filename_all = os.listdir(img_dir)
list_list = []
num = len(filename_all)
print num

# TODO Segmentation fault (core dumped)
n = 24
# for n thread
for i in range(n):
    n_num = num*1.0/n
    list_list.append(filename_all[int(np.ceil(n_num*i)):int(np.ceil(n_num*(i+1)))])
    print len(list_list[-1])

for i in range(n):
    processid = threading.Thread(target=vis_images, args=(str(i),list_list[i], img_dir,ano_dir, save_dir,))
    processid.start()
# vis_images('0',filename_all, img_dir,ano_dir, save_dir)
