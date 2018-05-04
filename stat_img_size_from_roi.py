import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# base_dir = '/home/wy/disks/disk0/datasets'
# db_str = 'DongSenDC/V4_mix6province_gray/plate/char'
base_dir = '/home/wy/disks/disk0/datasets'
db_str = 'NBPort/V3.2_gray_remixed/lorry/roi'
# db_str = 'GasStation/V0/plate/char'
#db_str = 'DongSenDC/V3/container/letter'
def get_img_dir(one_dir):
    return os.path.join(one_dir, 'data/JPEGImages')
def get_ano_dir(one_dir):
    return os.path.join(one_dir, 'data/Annotations')
ano_dir = get_ano_dir(os.path.join(base_dir, db_str))
height_width_list = []
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
for filename in os.listdir(ano_dir):
    #print filename
    charlist, charcoords = readAnoXML(os.path.join(ano_dir,filename))
    for i in range(len(charcoords)):
        coord = charcoords[i]
        (width, height) = (coord[2]-coord[0],coord[3]-coord[1])
        height_width_list.append(str(int(round(height))) + '_' + str(int(round(width))))

heights = [float(item.split('_')[0]) for item in height_width_list]
widths = [float(item.split('_')[1]) for item in height_width_list]
max_height = int(max(heights))
max_width = int(max(widths))
wh = np.zeros((max_height, max_width))
for item in height_width_list:
    [h, w] = item.split('_')
    wh[int(float(h)) - 1, int(float(w)) - 1] = wh[int(float(h)) - 1, int(float(w)) - 1] + 1

H = wh
fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H)
ax.set_aspect('equal')