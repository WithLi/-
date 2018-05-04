# -*- coding: utf-8 -*-
import numpy as np
import cv2, os
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import sys, shutil

reload(sys)
sys.setdefaultencoding('utf-8')


def get_img_dir(one_dir):
    return os.path.join(one_dir, 'data', 'JPEGImages')


def get_ano_dir(one_dir):
    return os.path.join(one_dir, 'data', 'Annotations')


def readAnoObject(filefpath_xml):
    tree = ET.parse(filefpath_xml)
    root = tree.getroot()
    return [obj.find('name').text for obj in root.findall('object')]

def checkMakeDir(one_dir):
    if os.path.exists(one_dir):
        shutil.rmtree(one_dir)
    os.makedirs(one_dir)


# read annotation xml file
# return number and coordinates
def readAnoXML(filefpath_xml):
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


def modifyAnoXML(filefpath_xml, coords, outfpath_xml):
    tree = ET.parse(filefpath_xml)
    root = tree.getroot()
    obj = root.findall('object')
    for i in range(len(obj)):
        coord = coords[i]
        bndbox = obj[i].find('bndbox')
        bndbox.find('xmin').text = str(int(coord[0]))
        bndbox.find('ymin').text = str(int(coord[1]))
        bndbox.find('xmax').text = str(int(coord[2]))
        bndbox.find('ymax').text = str(int(coord[3]))
    tree = ET.ElementTree(root)
    tree.write(outfpath_xml)

def modifyAnoXML2(filefpath_xml,im_shape, coords, outfpath_xml):
    tree = ET.parse(filefpath_xml)
    root = tree.getroot()
    # root.find('filename').text = outfpath_xml.strip().split('/')[-1][:7]+'.jpg'
    # print len(obj)
    # print np.asarray(coords).shape
    # tree = ET.parse(src_ano_fpath)
    # root = tree.getroot()
    filename = root.find('filename')
    filename.text = filefpath_xml.strip().split('/')[-1][:-3] + 'bmp'
    size = root.find('size')
    imshape = np.array([im_shape[1], im_shape[0], im_shape[2]])
    for i in range(len(size)):
        size[i].text = str(imshape[i])

    obj = root.findall('object')
    for i in range(len(obj)):
        # if obj[i].find('name').text not in ignore_list:
        #     coord = coords[i]
        #     bndbox = obj[i].find('bndbox')
        #     bndbox.find('xmin').text = str(int(coord[0]))
        #     bndbox.find('ymin').text = str(int(coord[1]))
        #     bndbox.find('xmax').text = str(int(coord[2]))
        #     bndbox.find('ymax').text = str(int(coord[3]))
        # else:
        #     root.remove(obj[i])
        coord = coords[i]
        bndbox = obj[i].find('bndbox')
        bndbox.find('xmin').text = str(int(coord[0]))
        bndbox.find('ymin').text = str(int(coord[1]))
        bndbox.find('xmax').text = str(int(coord[2]))
        bndbox.find('ymax').text = str(int(coord[3]))
    tree = ET.ElementTree(root)
    tree.write(outfpath_xml)

def whctrs(xyxy):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = xyxy[2] - xyxy[0] + 1
    h = xyxy[3] - xyxy[1] + 1
    x_ctr = xyxy[0] + 0.5 * (w - 1)
    y_ctr = xyxy[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def xyxy(whctrs):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    [w, h, x_ctr, y_ctr] = whctrs
    xyxy = [x_ctr - 0.5 * (w - 1),
            y_ctr - 0.5 * (h - 1),
            x_ctr + 0.5 * (w - 1),
            y_ctr + 0.5 * (h - 1)]
    return xyxy

################################generate xml file start########################################
def CreateObjInXML(root,num):
#    root = tree.getroot()
    example = ['word','Unspecified','1','0','\n\t\t\t']
    for i in range(num):
        example_obj = root.findall('object')[-1]
        example_obj.tail = '\n\t'
        obj = ET.SubElement(root,example_obj.tag)
        obj.text = '\n\t\t'
        for i in range(len(example_obj)):
            sub_obj = ET.SubElement(obj,example_obj[i].tag)
            sub_obj.text = example[i]
            if i == (len(example_obj)-1):
                sub_obj.tail = '\n\t'
            else:
                sub_obj.tail = '\n\t\t'
        example_obj = example_obj.find('bndbox')
        bndbox = obj.find('bndbox')
        for i in range(len(example_obj)):
            sub_obj = ET.SubElement(bndbox,example_obj[i].tag)
            sub_obj.text = 'xxxx'
            if i == (len(example_obj)-1):
                sub_obj.tail = '\n\t\t'
            else:
                sub_obj.tail = '\n\t\t\t'
    last_obj = root.findall('object')[-1]
    last_obj.tail = '\n'

def AssignValueToObj(root,allBoxes,label):
    objs = root.findall('object')
    for i in range(len(objs)):
        name = objs[i].find('name')
        name.text = label[i]
        bndbox = objs[i].find('bndbox')
        for j in range(len(bndbox)):
            bndbox[j].text = str(int(allBoxes[i,j])+1)

def gen_xml(image_shape,charlist, coords, out_ano_fpath, img_format):
    char_dict = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six',
                 '7': 'seven', '8': 'eight', '9': 'nine', 'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'E': 'e', 'F': 'f',
                 'G': 'g', 'H': 'h', 'I': 'i', 'J': 'j', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'O': 'o', 'P': 'p', 'Q': 'q',
                 'R': 'r', 'S': 's', 'T': 't', 'U': 'u', 'V': 'v', 'W': 'w', 'X': 'x', 'Y': 'y', 'Z': 'z'}
    src_xml_fpath = '/home/wy/Documents/workspace/analysis/code/basic/train/preprocessing_0/example.xml'
    tree = ET.parse(src_xml_fpath)
    root = tree.getroot()
    filename = root.find('filename')
    filename.text = out_ano_fpath.strip().split('/')[-1][:-3]+img_format
    size = root.find('size')
    imshape = np.array([image_shape[1], image_shape[0], image_shape[2]])
    for i in range(len(size)):
        size[i].text = str(imshape[i])
    boxes = np.asarray(coords)
    CreateObjInXML(root, boxes.shape[0] - 1)
    label = []
    for item in charlist:
        if item in char_dict.keys():
            label.append(char_dict[item])
        else:
            label.append(item)
    AssignValueToObj(root, boxes, label)
    tree.write(out_ano_fpath[:-3]+'xml')
################################generate xml file end########################################


print readAnoXML("/home/westwell/hj/hj_data/collect/all_1/char/error/data/Annotations/0bright.xml")