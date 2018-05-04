import os
import xml.etree.ElementTree as ET
from PIL import Image


xml_path = "/PATH/Annotations"
image_path = "/PATH/JPEGImages"
save_path = "PATH/sub/"
def readAnoXML(filefpath_xml):
    tree = ET.parse(filefpath_xml)
    root = tree.getroot()
    char_loca = {}
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        x1 = int(bndbox.find('xmin').text)
        y1 = int(bndbox.find('ymin').text)
        x2 = int(bndbox.find('xmax').text)
        y2 = int(bndbox.find('ymax').text)
        char_loca[obj.find('name').text] = [x1, y1, x2, y2]
    return char_loca



for p_xml in os.listdir(xml_path):
    img_pos = readAnoXML(os.path.join(xml_path,p_xml))
    for dis in img_pos:
        image_p = os.path.join(image_path,p_xml[:-4]+'.bmp')
        im = Image.open(image_p)
        position = img_pos[dis]
        im = im.crop((position[0], position[1], position[2], position[3]))
        sa_p = save_path+str(dis)+'/'
        print(sa_p)
        if not os.path.exists(os.path.dirname(sa_p)):
            os.makedirs(os.path.dirname(sa_p))
        im.save(sa_p+"/"+p_xml[:-4]+'.bmp')
