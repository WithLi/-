from skimage import exposure,img_as_float
import cv2
import os
import shutil
import xml.etree.ElementTree as ET

xml_path = "/home/westwell/hj/data/Annotations"
error_path = "/home/westwell/hj/hj_data/collect/all/error"
img_path = "/home/westwell/hj/data/JPEGImages"

def readAnoXML(xml_path,xml_name):
    tree = ET.parse(os.path.join(xml_path,xml_name))
    root = tree.getroot()
    charlist = []
    charcoord = []
    img_name = os.path.join(img_path,xml_name[:-4]+'.bmp')
    
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        x1 = int(bndbox.find('xmin').text)
        y1 = int(bndbox.find('ymin').text)
        x2 = int(bndbox.find('xmax').text)
        y2 = int(bndbox.find('ymax').text)
        charcoord.append([x1, y1, x2, y2])
        charlist.append(obj.find('name').text)

    if x1 > x2 or y1 > y2 or x1 < 0 or x2 < 0 or y1  < 0 or y2 < 0:
    	print("have error image or xml")
    	error_file_save(xml_path,img_path,error_path,xml_name,img_name)

    return charlist, charcoord
def error_file_save(xml_path,img_path,error_path,xml_name,img_name):
		shutil.move(os.path.join(xml_path,xml_name),os.path.join(error_path,xml_name))
		shutil.move(os.path.join(img_path,img_name),os.path.join(error_path,img_name))


for xml_name in os.listdir(xml_path):
	readAnoXML(xml_path,xml_name)



