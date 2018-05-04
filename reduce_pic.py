import os
import xml.etree.ElementTree as ET
import shutil
xml_path = "/home/westwell/hj/hj_data/CG/20171219/container/type/data/Annotations"
save_path = "/home/westwell/hj/hj_data/CG/20171219/container/type/data/test/JPEGImages"
save_path2 = "/home/westwell/hj/hj_data/CG/20171219/container/type/data/test/Annotations"
img_path = "/home/westwell/hj/hj_data/CG/20171219/container/type/data/JPEGImages"

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
        if obj.find('name').text == 'one' and a < 3000:
            shutil.move(os.path.join(os.path.join(img_path,xml_file[:-4]+'.bmp')),os.path.join(save_path,os.path.join(xml_file[:-4]+'.bmp')))
            shutil.move(filefpath_xml,os.path.join(save_path2,xml_file))
            a = a+1
            return 
        if obj.find('name').text == 'two' and b<3000:
            shutil.move(os.path.join(os.path.join(img_path,xml_file[:-4]+'.bmp')),os.path.join(save_path,os.path.join(xml_file[:-4]+'.bmp')))
            shutil.move(filefpath_xml,os.path.join(save_path2,xml_file))
            b = b+1
            return 
        if obj.find('name').text == 'four' and c<1000:
            shutil.move(os.path.join(os.path.join(img_path,xml_file[:-4]+'.bmp')),os.path.join(save_path,os.path.join(xml_file[:-4]+'.bmp')))
            shutil.move(filefpath_xml,os.path.join(save_path2,xml_file))
            c = c+1
            return 
        if obj.find('name').text == 'five' and d<1000:
            shutil.move(os.path.join(os.path.join(img_path,xml_file[:-4]+'.bmp')),os.path.join(save_path,os.path.join(xml_file[:-4]+'.bmp')))
            shutil.move(filefpath_xml,os.path.join(save_path2,xml_file))
            d = d+1
            return 
        if obj.find('name').text == 'g' and e < 3000:
            shutil.move(os.path.join(os.path.join(img_path,xml_file[:-4]+'.bmp')),os.path.join(save_path,os.path.join(xml_file[:-4]+'.bmp')))
            shutil.move(filefpath_xml,os.path.join(save_path2,xml_file))
            e = e+1
            return 
a=0
b=0
c=0
d=0
e=0
for xml_file in os.listdir(xml_path):
    #readAnoXML(os.path.join(xml_path,xml_file),xml_dict,xml_file)
    filefpath_xml = os.path.join(xml_path,xml_file)
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
        #xml_dict[obj.find('name').text] = xml_dict.get(obj.find('name').text,0)+1
        #print obj.find('name').text
        if obj.find('name').text == 'one' and a < 3000:
            shutil.move(os.path.join(os.path.join(img_path,xml_file[:-4]+'.jpg')),os.path.join(save_path,os.path.join(xml_file[:-4]+'.jpg')))
            shutil.move(filefpath_xml,os.path.join(save_path2,xml_file))
            a = a+1
            break
        elif obj.find('name').text == 'two' and b<3000:
            shutil.move(os.path.join(os.path.join(img_path,xml_file[:-4]+'.jpg')),os.path.join(save_path,os.path.join(xml_file[:-4]+'.jpg')))
            shutil.move(filefpath_xml,os.path.join(save_path2,xml_file))
            b = b+1
            break
        elif obj.find('name').text == 'four' and c<1000:
            shutil.move(os.path.join(os.path.join(img_path,xml_file[:-4]+'.jpg')),os.path.join(save_path,os.path.join(xml_file[:-4]+'.jpg')))
            shutil.move(filefpath_xml,os.path.join(save_path2,xml_file))
            c = c+1
            break
        elif obj.find('name').text == 'five' and d<000:
            shutil.move(os.path.join(os.path.join(img_path,xml_file[:-4]+'.jpg')),os.path.join(save_path,os.path.join(xml_file[:-4]+'.jpg')))
            shutil.move(filefpath_xml,os.path.join(save_path2,xml_file))
            d = d+1
            break
        elif obj.find('name').text == 'g' and e < 3000:
            shutil.move(os.path.join(os.path.join(img_path,xml_file[:-4]+'.jpg')),os.path.join(save_path,os.path.join(xml_file[:-4]+'.jpg')))
            shutil.move(filefpath_xml,os.path.join(save_path2,xml_file))
            e = e+1
            break
        else:
            pass
    print str(a)+"/"+str(b)+"/"+str(c)+"/"+str(d)+"/"+str(e)