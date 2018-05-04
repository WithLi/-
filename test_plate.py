# -*- coding: utf-8 -*-
import shutil, sys, argparse, os, cv2
from easydict import EasyDict as edict
import scipy.io as sio
import numpy as np
import xml.etree.ElementTree as ET

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path('/home/wy/Documents/workspace/analysis/code/basic')
sys.path.append('/home/westwell/hj/WellOean_train_script')
from cfgs.common import choose_dataset
CAFFE_PATH = os.path.join('/home/westwell/hj/WellOean_train_script/train/FRCNN_with_OHEM/caffes/caffe-fast-rcnn_FRCNN_add_roiAlign/python')
LIB_PATH = os.path.join('/home/westwell/hj/WellOean_train_script/train/FRCNN_with_OHEM/libs/lib_general')
add_path(CAFFE_PATH)
add_path(LIB_PATH)
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import caffe
import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='test faster rcnn model')
    parser.add_argument('--db_str', dest='db_str',
                        help='model type one of [SO, ST, SS, SF]',
                        default=0.3, type=str)
    parser.add_argument('--nt', dest='nms_thresh',
                        help='num threshold',
                        default=0.3, type=float)
    parser.add_argument('--ct', dest='conf_thresh',
                        help='confidence threshold',
                        default=0.5, type=float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class testModels(object):
    def __init__(self,save_dir):
        self.cfg = edict()
        self.NMS_THRESH = 0.3
        self.CONF_THRESH = 0.5
        self.save_dir = save_dir
        # self.choose_db(test_db_str)

    def choose_db(self, db_str='NBPort/V2/SO'):
        db_dict = choose_dataset('plate_zhakou/QZPort/char')
        if db_dict:
            self.element_list = db_dict.ELEMENT_LIST
            self.img_format = '.bmp'
        else:
            print 'error database string'
            sys.exit(0)

    def set_thresh(self, nms_thresh=0.3, conf_thresh=0.5):
        self.NMS_THRESH = nms_thresh
        self.CONF_THRESH = conf_thresh

    def vis_detections(self, fig, ax, im, class_name, dets, thresh, save_dir, im_name):
        # def vis_detections(self, im, class_name, dets, thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return
        dets = dets[inds, :]
        idx = np.argsort(dets[:, -1])
        for i in idx:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=1)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=7, color='white')
            img = Image.fromarray(im[bbox[1]:bbox[3],bbox[0]:bbox[2]])
            img.save(os.path.join(save_dir,'rois', im_name[:-4]+'_'+str(i)+'_'+str(class_name)+'.bmp'))
        ax.set_title(('{} detections with '
                      'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                      thresh),
                     fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        # plt.draw()
        # print os.path.join(save_dir, im_name[:-4] + '.jpg')
        fig.savefig(os.path.join(save_dir, im_name[:-4] + '.jpg'))
        plt.close(fig)

    def test_subimage(self, net, image_name):
        """Detect object classes in an image using pre-computed object proposals."""
        dataset = self.test_img_dir

        im_file = os.path.join(dataset, image_name)

        im = cv2.imread(im_file)
        # Visualize detections for each class
        CONF_THRESH = self.CONF_THRESH
        NMS_THRESH = self.NMS_THRESH
        pred_result = {}

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes, _ = im_detect(net, im)
        timer.toc()
        # print 'Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0])
        fig, ax = plt.subplots(figsize=(12, 12))
        im = im[:, :, (2, 1, 0)]
        ax.imshow(im, aspect='equal')
        for cls_ind, cls in enumerate(self.element_list):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            # print dets.shape
            # print "========="
            self.vis_detections(fig, ax, im, cls, dets, CONF_THRESH, self.save_dir, image_name)
            # inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            # dets = [dets[i, :] for i in inds]
            # if not len(inds) == 0:
            #     pred_result[cls] = dets

        # self.viewOnImage(fig, ax, im,  correct_classes, np.asarray(correct_coords),
        #                  self.result_data, image_name)

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

    def extract_sub_image(self, img_fpath, ano_fpath):
        im = cv2.imread(img_fpath)
        clslist, clscoords = self.readAnoXML(ano_fpath)
        digit_list = []
        for i in range(len(clslist)):
            cls = clslist[i]
            coord = clscoords[i]
            # print im.shape
            # print coord
            if cls in ['num']:
                digit_list.append(im[coord[1]:coord[3], coord[0]:coord[2], :])
        return digit_list
    def roi_test_plt(self, net, im, image_name, element_list):
        # Visualize detections for each class
        CONF_THRESH = self.CONF_THRESH
        NMS_THRESH = self.NMS_THRESH
        pred_result = {}

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)
        timer.toc()
        coords_list = []
        char_list = []
        fig, ax = plt.subplots(figsize=(12, 12))
        im = im.copy()[:, :, (2, 1, 0)]
        ax.imshow(im, aspect='equal')
        for cls_ind, cls in enumerate(element_list):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) != 0:
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    coords_list.append(bbox)
                    char_list.append(im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :])
                    ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]),
                                      bbox[2] - bbox[0],
                                      bbox[3] - bbox[1], fill=False,
                                      edgecolor='red', linewidth=1)
                    )
                    ax.text(bbox[0], bbox[1] - 2,
                            '{:s} {:.3f}'.format(cls, score),
                            bbox=dict(facecolor='blue', alpha=0.5),
                            fontsize=7, color='white')
        plt.axis('off')
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'image', image_name))
        plt.close(fig)
        return char_list
    def roi_test(self, net, im, image_name, element_list):
        # Visualize detections for each class
        CONF_THRESH = self.CONF_THRESH
        NMS_THRESH = self.NMS_THRESH
        pred_result = {}

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)
        image_copy = im.copy()
        timer.toc()
        coords_list = []
        char_list = []
        for cls_ind, cls in enumerate(element_list):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) != 0:
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    coords_list.append(bbox)
                    char_list.append(im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :])
                    cv2.rectangle(image_copy,(bbox[0], bbox[1]),(bbox[2],bbox[3]),(255,0,0), 1)
                    cv2.putText(image_copy, '{:s} {:.3f}'.format(cls, score), (int(bbox[0]), int(bbox[1] - 2)), \
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 155), 2, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(self.save_dir, 'image', image_name),image_copy)
        return char_list

    def general_test(self, net, im,image_name,j, element_list):
        # Visualize detections for each class
        CONF_THRESH = self.CONF_THRESH
        NMS_THRESH = self.NMS_THRESH
        pred_result = {}

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)
        timer.toc()
        cls_list = []
        coords_list = []
        coords_list_sum = []
        fig, ax = plt.subplots(figsize=(12, 12))
        im = im.copy()[:, :, (2, 1, 0)]
        ax.imshow(im, aspect='equal')
        for cls_ind, cls in enumerate(element_list):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) != 0:
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    coords_list.append(bbox)
                    cls_list.append(cls)
                    coords_list_sum.append(np.sum(bbox))
                    ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]),
                                      bbox[2] - bbox[0],
                                      bbox[3] - bbox[1], fill=False,
                                      edgecolor='red', linewidth=1)
                    )
                    ax.text(bbox[0], bbox[1] - 2,
                            '{:s} {:.3f}'.format(cls, score),
                            bbox=dict(facecolor='blue', alpha=0.5),
                            fontsize=7, color='white')
        plt.axis('off')
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'rois_result', image_name[:-4]+'_'+str(j) + '.jpg'))
        plt.close(fig)
        coords_list_sum_argsorted = sorted(range(len(coords_list_sum)), key=coords_list_sum.__getitem__)
        return [cls_list[i] for i in coords_list_sum_argsorted]
if __name__ == "__main__":

    # roi_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/DongSenDC/V2/plate/roi/roi_ohem_0928_dsv4_mix2rowplate_qz0923.caffemodel'
    # roi_pt = '/home/wy/disks/disk0/models/analysis_models/caffemodel/DongSenDC/V2/plate/roi/faster_rcnn_test.pt'
    # # digit_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/QZPort/V1/plate/char/char_ohem_0920_dsv4_qz0918_mix29p_dup1.caffemodel'
    # digit_model = '/home/wy/disks/disk0/models/analysis_models/caffemodel/DongSenDC/V2/plate/char_new/char_ohem_1005_qz0923_mixall.caffemodel'
    # digit_pt = '/home/wy/disks/disk0/models/analysis_models/caffemodel/DongSenDC/V2/plate/char_new/faster_rcnn_test.pt'



    # # yijiayou
    # roi_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/V0/plate/roi/ZF_512/roi_ohem_1116_gs_v0_addyijiayou.caffemodel'
    # roi_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/V0/plate/roi/ZF_512/faster_rcnn_test_original.pt'
    # # digit_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/QZPort/V1/plate/char/char_ohem_0920_dsv4_qz0918_mix29p_dup1.caffemodel'
    # digit_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/V0/plate/char/ZF_512/char_ohem_1115_addrotation_mixyijiayou.caffemodel'
    # digit_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/V0/plate/char/ZF_512/faster_rcnn_test_512.pt'

    # chexiangwang
    #roi_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/z_171121_chexiangwang/plate/roi/roi_ohem_1121_gs_v0_addchexiangwang.caffemodel'
    #roi_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/z_171121_chexiangwang/plate/roi/faster_rcnn_test.pt'



    # roi_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/roi_ohem_1122_pg_all_roi.caffemodel'
    # roi_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/faster_rcnn_test.pt'
    # # digit_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/QZPort/V1/plate/char/char_ohem_0920_dsv4_qz0918_mix29p_dup1.caffemodel'
    # digit_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/yijiayou/plate/char/char_ohem_1115_addrotation.caffemodel'
    # digit_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/yijiayou/plate/char/faster_rcnn_test.pt'


    #qingdao
    # roi_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/QingDao/test/plate/roi/roi_ohem_1028_qd.caffemodel'
    # roi_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/QingDao/test/plate/roi/faster_rcnn_test.pt'
    # digit_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/DongSenDC/V4_mixColor/plate/char/char_last.caffemodel'
    # digit_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/DongSenDC/V4_mixColor/plate/char/faster_rcnn_test.pt'

    # # yijiayou
    # roi_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/yijiayou/plate/roi/roi_ohem_180108_plate_roi.caffemodel'
    # roi_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/yijiayou/plate/roi/faster_rcnn_test.pt'
    # # digit_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/QZPort/V1/plate/char/char_ohem_0920_dsv4_qz0918_mix29p_dup1.caffemodel'
    # digit_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/yijiayou/plate/char/char_ohem_180109_yijiayou_mixpg_512.caffemodel'
    # digit_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/GasStation/yijiayou/plate/char/faster_rcnn_test_512.pt'
    # # digit_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/zimaoqu/sh_z_171214_plate_fixano/plate/char/char_ohem_171214_zimaoqu_plate.caffemodel'
    # # digit_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/zimaoqu/sh_z_171214_plate_fixano/plate/char/faster_rcnn_test.pt'

    # yantai
    # roi_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/plate_zhakou/YanTai/test/roi/roi_ohem_180116_yt_plate_roi.caffemodel'
    # roi_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/plate_zhakou/YanTai/test/roi/faster_rcnn_test.pt'
    # digit_model = '/home/wy/Documents/workspace/analysis/model/caffemodel/plate_zhakou/YanTai/test/char/char_ohem_180116_yt_mixpg.caffemodel'
    # digit_pt = '/home/wy/Documents/workspace/analysis/model/caffemodel/plate_zhakou/YanTai/test/char/faster_rcnn_test_addgang.pt'

    # yantian
    # roi_model = '/home/wy/disks/disk0/models/analysis_models/caffemodel/plate_zhakou/YanTian/test/roi/roi_ohem_180126_yantian_plate_roi.caffemodel'
    # roi_pt = '/home/wy/disks/disk0/models/analysis_models/caffemodel/plate_zhakou/YanTian/test/roi/faster_rcnn_test.pt'
    # digit_model = '/home/wy/disks/disk0/models/analysis_models/caffemodel/plate_zhakou/YanTian/test/char/without_yan/char_ohem_180209_yantian_plate.caffemodel'
    # digit_pt = '/home/wy/disks/disk0/models/analysis_models/caffemodel/plate_zhakou/YanTian/test/char/without_yan/faster_rcnn_test_512.pt'

    # qzport
    roi_model = '/home/wy/Desktop/decrypt/PlateRecognitionModel/plate_0914_gray.caffemodel'
    roi_pt = '/home/wy/Desktop/decrypt/PlateRecognitionModel/faster_rcnn_test.pt'
    digit_model = '/home/westwell/hj/WellOean_train_script/train/FRCNN_with_OHEM/code/180410/plate_zhakou_QZPort_char_180428.caffemodel'
    digit_pt = '/home/westwell/hj/WellOean_train_script/train/FRCNN_with_OHEM/model/plate_zhakou_QZPort_char/faster_rcnn_alt_opt/faster_rcnn_test.pt'




    # test_img_dir = '/home/wy/disks/disk1/test/test_image'
    # save_dir = '/home/wy/disks/disk1/test/test_result'
    # db_str = 'GasStation/chexiangwang/171116/fc5ee0dd-0bb8-46e1-8d19-0c700890abf8_9F2E717605207_3_20171029_103735_120610.mp4_20171116_163913'
    # test_img_dir = os.path.join(test_img_dir, db_str)
    # save_dir = os.path.join(save_dir, db_str)
    test_img_dir = '/home/westwell/wy/test_video/zhakou/cv/test_videos/qinzhou/video/j3/save_test/GX_N85767'
    save_dir = test_img_dir + '_reg'

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, 'final_result'))
        os.makedirs(os.path.join(save_dir, 'rois'))
        os.makedirs(os.path.join(save_dir, 'image'))
        os.makedirs(os.path.join(save_dir, 'rois_result'))
    #
    # else:
    #     shutil.rmtree(save_dir)
    roi_element_list = ['plate']
    # char_element_list = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    #                                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's',
    #                                't', 'u', 'v', 'w', 'x', 'y', 'z','sh', 'js', 'ha', 'ah', 'zj', 'bj', 'tj', 'cq',
    #                                 'he', 'yn', 'ln', 'hl', 'hn', 'sd','sc', 'jx', 'hb', 'gs',
    #                                'sx', 'sn', 'jl', 'fj', 'gz', 'gd', 'gx', 'qh', 'hi', 'nx', 'xz', 'nm', 'xj']
    char_element_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', \
                         'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                         'SH', 'JS', 'HA', 'AH', 'ZJ', 'BJ', 'TJ', 'CQ','HE', 'YN', 'LN', 'HL', 'HN', 'SD','SC', 'JX',
                         'HB', 'GS', 'SX', 'SN', 'JL', 'FJ', 'GZ', 'GD', 'GX', 'QH', 'HI', 'NX', 'XZ', 'NM', 'XJ', 'factory','square','inside']
                         #'XUE','GUA', 'JING', 'SHI', 'LING', 'JUN']
    # char_element_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', \
    #            'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', \
    #            "沪", "苏", "豫", "皖", "浙", "京", "津", "渝", "冀", "云", "辽", "黑", "湘", "鲁", "川", "赣", "鄂", "甘", "晋", \
    #            "陕", "吉", "闽", "贵", "粤", "桂", "青", "琼", "宁", "藏", "蒙", "新", "学", "挂",  "警",  "使", "领", "军"]
    # char_element_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', \
    #            'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', \
    #            "桂",'厂','场','内']
    # char_dict = {'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7',
    #  'eight': '8', 'nine': '9', 'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G',
    #  'h': 'H', 'i': 'I', 'j': 'J', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O', 'p': 'P',
    #  'q': 'Q', 'r': 'R', 's': 'S', 't': 'T', 'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y', 'z': 'Z'}

    cfg.TEST.HAS_RPN = True
    # caffe.set_mode_cpu()
    cfg_analysis.TEST.GPU_MODE = 1
    if cfg_analysis.TEST.GPU_MODE == 1:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        caffe.set_device(0)
        cfg.GPU_ID = 0
    else:
        caffe.set_mode_cpu()
    cfg.TEST.RPN_PRE_NMS_TOP_N = 6000
    cfg.TEST.RPN_POST_NMS_TOP_N = 300

    tm = testModels(save_dir)
    nms_thresh = 0.3
    conf_thresh = 0.5
    tm.set_thresh(nms_thresh=nms_thresh, conf_thresh=conf_thresh)

    roi_net = caffe.Net(roi_pt, roi_model, caffe.TEST)
    digit_net = caffe.Net(digit_pt, digit_model, caffe.TEST)

    im_names = os.listdir(test_img_dir)
    # im_names = sorted(im_names, key=lambda x: int(x[:-4]))
    im_names = im_names
    # im_names = ['18444.jpg']
    # print len(im_names)

    # bar = progressbar.ProgressBar()

    out_fpath = os.path.join(save_dir, 'pred_result.txt')

    fh = open(out_fpath, 'w+')
    for i in range(len(im_names)):
    # for i in bar(range(len(im_names)))[:10]:
        img_fpath = os.path.join(test_img_dir, im_names[i])
        # ano_fpath = os.path.join(test_ano_dir, im_names[i][:-4]+'.xml')
        img = cv2.imread(img_fpath)
        im = img
        im[:,:,0] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im[:,:,1] = im[:,:,0]
        im[:,:,2] = im[:,:,0]
        char_result = []
        plate_list = tm.roi_test(roi_net, im,im_names[i].strip(), roi_element_list)
        for j in range(len(plate_list)):
            item = plate_list[j]
            # print os.path.join(save_dir, im_names[i][:-4]+'_'+str(j)+'.bmp')
            cv2.imwrite(os.path.join(save_dir, 'rois',im_names[i][:-4]+'_'+str(j)+'.bmp'),item)
            char_result.append(tm.general_test(digit_net, item,im_names[i],j, char_element_list))
        char_str = ''
        if len(char_result) != 0:
            for j in range(len(char_result)):
                for k in range(len(char_result[j])):
                    char_result[j][k] = char_result[j][k]
                char_str = char_str+' '+''.join(char_result[j])
        cv2.imwrite(os.path.join(save_dir,'final_result',im_names[i][:-4]+'_'+char_str+'.jpg'), cv2.imread(img_fpath))
        fh.write(' '.join([im_names[i],char_str])+'\n')
    fh.close()
    print "Done"