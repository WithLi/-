import os
import xml.etree.ElementTree as ET
import shutil
xml_path = "/home/westwell/hj/hj_data/CG/20171219/container/type/data/Annotations"
save_path = "/home/westwell/hj/hj_data/CG/20171219/container/type/data/some_aug_pic/JPEGImages"
save_path2 = "/home/westwell/hj/hj_data/CG/20171219/container/type/data/some_aug_pic/Annotations"
img_path = "/home/westwell/hj/hj_data/CG/20171219/container/type/data/JPEGImages"

xml_dict = {}
def readAnoXML(filefpath_xml,xml_dict,xml_file):
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
        #charcoord.append([x1, y1, x2, y2])
        #charlist.append(obj.find('name').text)
        xml_dict[obj.find('name').text] = xml_dict.get(obj.find('name').text,0)+1
        #print obj.find('name').text
        if obj.find('name').text == 'zero' or obj.find('name').text == 'six' or obj.find('name').text == 'seven' or obj.find('name').text == 'e' or obj.find('name').text == 'k' or obj.find('name').text == 'l' or obj.find('name').text == 't':
            shutil.copy(os.path.join(os.path.join(img_path,xml_file[:-4]+'.jpg')),os.path.join(save_path,os.path.join(xml_file[:-4]+'.jpg')))
            shutil.copy(filefpath_xml,os.path.join(save_path2,xml_file))
            print 'a'
    return xml_dict

a = 0
for xml_file in os.listdir(xml_path):
    readAnoXML(os.path.join(xml_path,xml_file),xml_dict,xml_file)
    print a
    a=a+1

for key in xml_dict:
    print key + ":" + str(xml_dict[key])