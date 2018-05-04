import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# base_dir = '/home/wy/disks/disk0/datasets'
# db_str = 'DongSenDC/V4_mix6province_gray/plate/char'
base_dir = '/home/wy/disks/disk0/datasets'
db_str = 'plate/GasStation/storage/171129_pg_addR_addPF_blur/char'
#db_str = 'GasStation/z_171110_yijiayou/plate/char'
#db_str = 'DongSenDC/V3/container/letter'
def get_img_dir(one_dir):
    return os.path.join(one_dir, 'data/JPEGImages')

img_dir = get_img_dir(os.path.join(base_dir, db_str))
height_width_list = []
for filename in os.listdir(img_dir):
    #print filename
    im = Image.open(os.path.join(img_dir,filename))
    (width, height) = im.size
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