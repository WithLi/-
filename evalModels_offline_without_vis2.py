import shutil, sys, argparse, os, cv2
from easydict import EasyDict as edict
import scipy.io as sio
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from shutil import copy2

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path('/home/wy/Documents/workspace/analysis/code/basic')
from cfgs.cfg_analysis_caffe import cfg_analysis,choose_dataset

add_path(cfg_analysis.FASTER_RCNN.CAFFE_PATH)
add_path(cfg_analysis.FASTER_RCNN.LIB_PATH)
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import caffe


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='test faster rcnn model')
    parser.add_argument('--db_str', dest='db_str',
                        help='model type one of NBPort/somedate/container/roi',
                        default='NBPort/tmp/container/roi', type=str)
    parser.add_argument('--nt', dest='nms_thresh',
                        help='num threshold',
                        default=0.3, type=float)
    parser.add_argument('--ct', dest='conf_thresh',
                        help='confidence threshold',
                        default=0.5, type=float)
    parser.add_argument('--ot', dest='overlap_thresh',
                        help='overlap threshold',
                        default=0.5, type=float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class evalModels(object):
    def __init__(self, mat_fpath, db_str, test_img_dir, test_ano_dir, test_result_dir):
        self.cfg = edict()
        self.cfg.mat_fpath = mat_fpath
        self.test_img_dir = test_img_dir
        self.test_ano_dir = test_ano_dir
        self.result_data = os.path.join(test_result_dir, 'data')
        self.stat_count = {}
        self.stat_total = {}
        self.NMS_THRESH = 0.3
        self.CONF_THRESH = 0.5
        self.OVERLAP_THRESH = 0.5
        self.CLASSES = ()
        self.pred_result = []
        self.choose_db(db_str)
        self.missing_image_list = []
        self.incorrect_image_num = 0

    def choose_db(self, db_str='NBPort/V2/SO'):
        db_dict = choose_dataset(db_str)
        if db_dict:
            self.element_list = db_dict.ELEMENT_LIST
            self.element_dict = db_dict.ELEMENT_DICT
            self.img_format = 'jpg'
            for cls in self.element_list:
                self.stat_count[cls] = 0
                self.stat_total[cls] = 0
        else:
            print 'error database string'
            sys.exit(0)


    def set_thresh(self, nms_thresh=0.3, conf_thresh=0.5, overlap_thresh=0.5):
        self.NMS_THRESH = nms_thresh
        self.CONF_THRESH = conf_thresh
        self.OVERLAP_THRESH = overlap_thresh

    def read_test_result(self, mat_fpath=None):
        if mat_fpath is None:
            mat_fpath = self.cfg.mat_fpath

        mat_file = sio.loadmat(mat_fpath)
        test_result = mat_file['pred_result'][0]
        pred_result = {}
        for i in range(len(test_result)):
            # print i
            im_name = test_result[i][0, 0]['name'][0]
            # print im_name
            tmp = test_result[i][0, 0]['result']
            # print len(tmp[0, 0])
            tmp_type = list(tmp.dtype.names)
            # print tmp_type
            single_pred_result = {}
            for j in range(len(tmp_type)):
                single_pred_result[tmp_type[j]] = tmp[tmp_type[j]][0, 0]
            pred_result[im_name] = single_pred_result
        return pred_result

    def readAnoXML(self, filefpath_xml):
        tree = ET.parse(filefpath_xml)
        root = tree.getroot()
        charlist = []
        charcoord = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            x1 = int(bndbox.find('xmin').text)
            y1 = int(bndbox.find('ymin').text)
            x2 = int(bndbox.find('xmax').text)
            y2 = int(bndbox.find('ymax').text)
            charcoord.append([x1, y1, x2, y2])
            charlist.append(obj.find('name').text)
        return charlist, charcoord

    def bbox_overlap(self, g_bbox, p_bbox):
        area1 = (g_bbox[3] - g_bbox[1]) * (g_bbox[2] - g_bbox[0])
        area2 = (p_bbox[3] - p_bbox[1]) * (p_bbox[2] - p_bbox[0])
        # calculate overlap section's coordinate(top left and bottom right)
        xx1 = np.maximum(g_bbox[0], p_bbox[0])
        yy1 = np.maximum(g_bbox[1], p_bbox[1])
        xx2 = np.minimum(g_bbox[2], p_bbox[2])
        yy2 = np.minimum(g_bbox[3], p_bbox[3])
        # width
        w = np.maximum(0.0, xx2 - xx1 + 1)
        # height
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        overlap = inter * 1.0 / (area1 + area2 - inter)
        return overlap

    def eval_bbox(self, charlist, charcoord, pred_result):
        correct_result = []
        correct_classes = []
        for i in range(len(charlist)):
            char = charlist[i]
            g_bbox = charcoord[i]
            if char in pred_result:
                dets = pred_result[char]
                overarea = [self.bbox_overlap(g_bbox, det[:4]) for det in dets]
                inds = np.where(np.asarray(overarea) > self.OVERLAP_THRESH)[0]
                # print inds
                # inds = np.argsort(overarea)
                # if inds.shape[0] > 3:
                #     inds = inds[:3]
                # # print list(inds)
                # keep = dets[list(inds), :]
                # inds = [np.argmax(keep[:, -1])]
                if len(inds) == 0:
                    correct_result.append([0,0,0,0,0])
                    correct_classes.append(' ')
                else:
                    correct_result.append(dets[inds[0]])
                    correct_classes.append(char)
            else:
                correct_result.append([0,0,0,0,0])
                correct_classes.append(' ')

        return correct_classes, correct_result


    def eval_image(self, image_name, preds):
        """Detect object classes in an image using pre-computed object proposals."""
        # Load the demo image
        filefpath_xml = os.path.join(self.test_ano_dir, image_name + '.xml')
        if not os.path.exists(filefpath_xml):
            return
        charlist, charcoord = self.readAnoXML(filefpath_xml)

        CONF_THRESH = self.CONF_THRESH
        NMS_THRESH = self.NMS_THRESH
        pred_result = {}
        dets_all = []
        for cls, dets in preds.iteritems():
            # print dets.shape
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            # print dets.shape
            # print "========="
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if not len(inds) == 0:
                dets = [dets[i, :] for i in inds]
                if dets_all == [] :
                    dets_all = np.concatenate((np.asarray(dets), np.ones((len(dets),1))*self.element_dict[cls]),axis=1)
                else:
                    dets_all = np.concatenate((dets_all,np.concatenate((np.asarray(dets), np.ones((len(dets),1))*self.element_dict[cls]),axis=1)),axis=0)
                # pred_result[cls] = np.asarray(dets)
        if dets_all != []:
            keep = nms(dets_all[:,:-1].astype(np.float32), NMS_THRESH)
            dets_all = dets_all[keep, :]
            # print dets_all
            for i in range(len(self.element_list)):
                dets =dets_all[np.where(dets_all[:,-1]==i)[0],:-1].tolist()
                if len(dets) != 0:
                    pred_result[self.element_list[i]] = dets
        # print pred_result
        # fig, ax = plt.subplots(figsize=(12, 12))
        ########################## for generalize roi##########################
        # tmp_charlist = []
        # tmp_coords = []
        # for i in range(len(charlist)):
        #     if charlist[i] not in ['foot','numhsq']:
        #         tmp_charlist.append(charlist[i])
        #         tmp_coords.append(charcoord[i])
        # charlist = tmp_charlist
        # charcoord = tmp_coords
        ##############################end########################################
        i = 0
        while i < len(charlist):
            if charlist[i] not in self.element_list:
                del(charlist[i])
                del(charcoord[i])
            else:
                i += 1
        correct_classes, correct_coords = self.eval_bbox(charlist, charcoord, pred_result)
        if correct_classes != charlist:
            self.incorrect_image_num = self.incorrect_image_num + 1
        for i in range(len(charlist)):
            # print charlist, i, charlist[i]
            self.stat_total[charlist[i]] += 1
            if not correct_classes[i] == ' ':
                self.stat_count[correct_classes[i]] += 1

if __name__ == "__main__":
    args = parse_args()
    print('Called with args:')
    print(args)
    # db_str = 'SF'
    db_str = args.db_str
    test_base_dir = cfg_analysis.EVAL.DATA_BASE_DIR
    test_img_suffix = 'data/JPEGImages'
    test_ano_suffix = 'data/Annotations'
    test_img_dir = os.path.join(test_base_dir, db_str, test_img_suffix)
    test_ano_dir = os.path.join(test_base_dir, db_str, test_ano_suffix)

    test_result_dir = os.path.join(cfg_analysis.EVAL.RESULT_DIR, db_str)
    mat_file = db_str.strip().split('/')[-1] + "_result.mat"
    mat_fpath = os.path.join(test_result_dir, mat_file)
    result_dir = os.path.join(test_result_dir, 'data')

    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        for single_file in os.listdir(result_dir):
            os.unlink(os.path.join(result_dir, single_file))

    em = evalModels(mat_fpath, db_str, test_img_dir, test_ano_dir, test_result_dir)

    pred_result = em.read_test_result()

    nms_thresh = args.nms_thresh
    overlap_thresh = args.overlap_thresh
    conf_thresh = args.conf_thresh
    em.set_thresh(nms_thresh=nms_thresh, conf_thresh=conf_thresh, overlap_thresh=overlap_thresh)

    image_num = 0
    for k, v in pred_result.iteritems():
        # print k
        image_num = image_num + 1
        em.eval_image(image_name=k, preds=v)
        # break

    stat_count = em.stat_count
    stat_total = em.stat_total

    total_correct = 0
    total_num = 0
    for k in em.element_list:
        if k not in stat_count.keys():
            continue
        v = stat_count[k]
        if stat_total[k] != 0:
            print k, stat_count[k], stat_total[k], stat_count[k] * 1.0 / stat_total[k]
            total_correct += stat_count[k]
            total_num += stat_total[k]
    incorrect_image_num = em.incorrect_image_num
    if total_num != 0:
        print total_correct, total_num, total_correct * 1.0 / total_num
        print "whole image stat:"
        print image_num - incorrect_image_num, image_num, (image_num - incorrect_image_num) * 1.0 / image_num
    print "Done"
