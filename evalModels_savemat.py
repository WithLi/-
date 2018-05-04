#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys, cv2
import argparse
import xml.etree.ElementTree as ET
import sys
import progressbar
reload(sys)
sys.setdefaultencoding('utf-8')

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path('/home/wy/Documents/workspace/analysis/code/basic')
from cfgs.cfg_analysis_caffe import cfg_analysis, choose_dataset
# for faster rcnn
add_path(cfg_analysis.FASTER_RCNN.CAFFE_PATH)
add_path(cfg_analysis.FASTER_RCNN.LIB_PATH)

#for rfcn
#add_path(cfg_analysis.RFCN.CAFFE_PATH)
#add_path(cfg_analysis.RFCN.LIB_PATH)
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import caffe

caffe.__dict__['flag']='what'
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='test faster rcnn model')
    parser.add_argument('--db_str', dest='db_str',
                        help='example: NBPort/V1/SO',
                        default=0.3, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class test_ult(object):
    def __init__(self, test_prototxt, test_caffemodel, test_img_dir, test_ano_dir, test_result_dir):
        self.prototxt = test_prototxt
        self.caffemodel = test_caffemodel
        self.test_img_dir = test_img_dir
        self.test_ano_dir = test_ano_dir
        self.result_dir = test_result_dir
        self.pred_result = []

    def choose_db(self, db_str='NBPort/V2/SO'):
        db_dict = choose_dataset(db_str)
        if db_dict:
            print db_dict
            self.element_list = db_dict.ELEMENT_LIST
            self.img_format = 'jpg'
        else:
            print 'error database string'
            sys.exit(0)

    def eval_image(self, net, image_name):
        """Detect object classes in an image using pre-computed object proposals."""

        # Load the demo image
        filefpath_xml = os.path.join(self.test_ano_dir, image_name[:-4] + '.xml')
        if not os.path.exists(filefpath_xml):
            print "missing annotation: ", filefpath_xml
            return

        im_file = os.path.join(self.test_img_dir, image_name)

        img = cv2.imread(im_file)
        im = img
        im[:, :, 0] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im[:, :, 1] = im[:, :, 0]
        im[:, :, 2] = im[:, :, 0]
        # print im_file
        if not self.valid_img_size(im):
            return
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)# for faster rcnn
        #scores, boxes = im_detect(net, im) # for rfcn
        timer.toc()
        # print image_name[:-4]
        # print ('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0])

        single_result_all = {}
        single_result_all["name"] = image_name[:-4]
        single_result_all["result"] = {}
        for cls_ind, cls in enumerate(self.element_list):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)] #for faster rcnn
            # cls_boxes = boxes[:, 4 : 8] # rfcn
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            single_result_all["result"][cls] = dets
        self.pred_result.append(single_result_all)

    def valid_img_size(self, img):
        try:
            w, h, ch = img.shape
            s = min(w, h)
            l = max(w, h)
            ratio = min(600.0 / s, 1000.0 / l)
            # print w, h, ratio
            if (w * ratio / 16 < 6) or (h * ratio / 16 < 6) or (w <= 16) or (h <= 16):
                return False
            else:
                return True
        except:
            return False


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    db_str = args.db_str  # db_str = 'SF'
    test_prototxt = os.path.join(cfg_analysis.EVAL.PROTOTXT_BASE_DIR, db_str, 'faster_rcnn_test.pt') #for faster rcnn
    # test_prototxt = os.path.join(cfg_analysis.EVAL.PROTOTXT_BASE_DIR, db_str, 'test_agnostic.prototxt') #for rfcn
    test_caffemodel = os.path.join(cfg_analysis.EVAL.MODEL_BASE_DIR, db_str, db_str.strip().split('/')[-1] + '_last.caffemodel')
    #test_caffemodel = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/yijiayou/plate/char/char_ohem_1115_addrotation.caffemodel'
    # faster_rcnn_version = 'ohem'

    test_base_dir = cfg_analysis.EVAL.DATA_BASE_DIR
    test_img_suffix = 'data/JPEGImages'
    test_ano_suffix = 'data/Annotations'
    test_img_dir = os.path.join(test_base_dir, db_str, test_img_suffix)
    test_ano_dir = os.path.join(test_base_dir, db_str, test_ano_suffix)

    test_result_dir = os.path.join(cfg_analysis.EVAL.RESULT_DIR, db_str)

    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    ######################
    # caffe config
    ######################

    if not os.path.isfile(test_caffemodel):
        raise IOError(('{:s} not found.\n').format(test_caffemodel))

    cfg.TEST.HAS_RPN = True
    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    cfg.TEST.RPN_PRE_NMS_TOP_N = 6000
    cfg.TEST.RPN_POST_NMS_TOP_N = 300

    test_obj = test_ult(test_prototxt, test_caffemodel, test_img_dir, test_ano_dir, test_result_dir)
    test_obj.choose_db(db_str)
    timer = Timer()
    timer.tic()
    net = caffe.Net(test_prototxt, test_caffemodel, caffe.TEST)
    net.__dict__['what']=None
    timer.toc()

    print '\n\nLoaded network {:s} for {:.3f}s'.format(test_caffemodel, timer.total_time)

    # Warmup on a dummy image
    # im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _ = im_detect(net, im)
    print test_obj.element_list
    print len(test_obj.element_list)
    im_names = os.listdir(test_img_dir)
    print "test image number: ",len(im_names)
    pb = progressbar.ProgressBar()
    for i in pb(range(len(im_names))):
        im_name = im_names[i]
        # print im_name
        test_obj.eval_image(net, im_name)
    pred_result = test_obj.pred_result

    sio.savemat(os.path.join(test_result_dir, db_str.strip().split('/')[-1] + '_result.mat'), {"pred_result": pred_result})
    print "save image to: ", os.path.join(test_result_dir, db_str.strip().split('/')[-1] + '_result.mat')
    print "Done"

