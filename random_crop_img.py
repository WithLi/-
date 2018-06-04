from PIL import Image
import os
import xml.etree.ElementTree as ET
import random
import xml.dom.minidom
import cv2
xml_path = ""
image_path = ""

def pascal_voc_xml_trans(xml_filename,folder,filename,source_database,source_annotation,source_image,width,height,depth,segmented,name,pose,truncated,difficult,object_position):
    # root
    object_size = len(object_position)
    impl = xml.dom.minidom.getDOMImplementation()
    dom = impl.createDocument(None, 'annotation', None)
    node_annotation = dom.documentElement
    # folder
    node_folder = dom.createElement('folder')
    text_folder = dom.createTextNode(folder)
    node_folder.appendChild(text_folder)
    node_annotation.appendChild(node_folder)
    # filename
    node_filename = dom.createElement('filename')
    text_filename=dom.createTextNode(filename)
    node_filename.appendChild(text_filename)
    node_annotation.appendChild(node_filename)
    # source
    node_source=dom.createElement('source')
    node_source_database=dom.createElement('database')
    text_source_database=dom.createTextNode(source_database)
    node_source_database.appendChild(text_source_database)
    node_source_annotation=dom.createElement('annotation')
    text_source_annotation=dom.createTextNode(source_annotation)
    node_source_annotation.appendChild(text_source_annotation)
    node_source_image=dom.createElement('image')
    text_source_image=dom.createTextNode(source_image)
    node_source_image.appendChild(text_source_image)
    node_source.appendChild(node_source_database)
    node_source.appendChild(node_source_annotation)
    node_source.appendChild(node_source_image)
    node_annotation.appendChild(node_source)
    # size
    node_size=dom.createElement('size')
    node_size_width=dom.createElement('width')
    text_size_width=dom.createTextNode(width)
    node_size_width.appendChild(text_size_width)
    node_size_height=dom.createElement('height')
    text_size_height=dom.createTextNode(height)
    node_size_height.appendChild(text_size_height)
    node_size_depth=dom.createElement('depth')
    text_size_depth=dom.createTextNode(depth)
    node_size_depth.appendChild(text_size_depth)
    node_size.appendChild(node_size_width)
    node_size.appendChild(node_size_height)
    node_size.appendChild(node_size_depth)
    node_annotation.appendChild(node_size)
    # segmented
    node_segmented=dom.createElement('segmented')
    text_segmented=dom.createTextNode(segmented)
    node_segmented.appendChild(text_segmented)
    node_annotation.appendChild(node_segmented)
    # object
    for i in range(object_size):
        node_object=dom.createElement('object')
        node_object_name=dom.createElement('name')
        text_object_name=dom.createTextNode(str(name[i]))
        node_object_name.appendChild(text_object_name)
        node_object_pose=dom.createElement('pose')
        text_object_pose=dom.createTextNode(pose)
        node_object_pose.appendChild(text_object_pose)
        node_object_truncated=dom.createElement('truncated')
        text_object_truncated=dom.createTextNode(truncated)
        node_object_truncated.appendChild(text_object_truncated)
        node_object_difficult=dom.createElement('difficult')
        text_object_difficult=dom.createTextNode(difficult)
        node_object_difficult.appendChild(text_object_difficult)
        # object-bndbox
        node_object_bndbox=dom.createElement('bndbox')
        node_object_bndbox_xmin=dom.createElement('xmin')
        node_object_bndbox_ymin=dom.createElement('ymin')
        node_object_bndbox_xmax=dom.createElement('xmax')
        node_object_bndbox_ymax=dom.createElement('ymax')
        text_object_bndbox_xmin=dom.createTextNode(str(object_position[i][0]))
        text_object_bndbox_ymin=dom.createTextNode(str(object_position[i][1]))
        text_object_bndbox_xmax=dom.createTextNode(str(object_position[i][2]))
        text_object_bndbox_ymax=dom.createTextNode(str(object_position[i][3]))
        node_object_bndbox_xmin.appendChild(text_object_bndbox_xmin)
        node_object_bndbox_ymin.appendChild(text_object_bndbox_ymin)
        node_object_bndbox_xmax.appendChild(text_object_bndbox_xmax)
        node_object_bndbox_ymax.appendChild(text_object_bndbox_ymax)
        node_object_bndbox.appendChild(node_object_bndbox_xmin)
        node_object_bndbox.appendChild(node_object_bndbox_ymin)
        node_object_bndbox.appendChild(node_object_bndbox_xmax)
        node_object_bndbox.appendChild(node_object_bndbox_ymax)
        node_object.appendChild(node_object_name)
        node_object.appendChild(node_object_pose)
        node_object.appendChild(node_object_truncated)
        node_object.appendChild(node_object_difficult)
        node_object.appendChild(node_object_bndbox)
        node_annotation.appendChild(node_object)

    pascal_voc_xml=open(xml_filename,'w+')
    content=node_annotation.toprettyxml()
    pascal_voc_xml.writelines(content)
    pascal_voc_xml.close()
def trans_function(data_dict,image_name):
    object_position = []
    object_name = []
    for bbox in data_dict:
        object_name.append(bbox)
        filename = bbox
        xmin=int(data_dict[bbox][0])
        ymin=int(data_dict[bbox][1])
        xmax=int(data_dict[bbox][2])
        ymax=int(data_dict[bbox][3])
        object_position.append([xmin,ymin,xmax,ymax])

    width = 1920
    height = 1080
    depth = 3
    xml_filename= image_name[:-4]+".xml"
    pascal_voc_xml_trans(xml_filename,'armor',filename,'armor','armor','armor',str(width),str(height),str(depth),'0',object_name,'Unspecified','0','0',object_position)


def readAnoXML(filefpath_xml):
    tree = ET.parse(filefpath_xml)
    root = tree.getroot()
    char_loca = {}
    all_object = []
    min1 = 1000
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        x1 = int(bndbox.find('xmin').text)
        y1 = int(bndbox.find('ymin').text)
        x2 = int(bndbox.find('xmax').text)
        y2 = int(bndbox.find('ymax').text)
        char_loca[obj.find('name').text] = [x1, y1, x2, y2]
        min2 = min(x1, y1, x2, y2)
        min1 = min(min1,min2)
    return char_loca,min1

def get_image_and_xml(img_p,xml_p):
    img=Image.open(img_p)  #打开图像
    position_dict,min_pic = readAnoXML(xml_p)
    top_position = position_dict
    left_position = position_dict
    right_position = position_dict
    botton_position = position_dict
    size = img.size
    print(position_dict)
    a = random.randint(0,min_pic)
    top_box=(0, a, size[0], size[1])  #crop top
    top_position = every_position_crop(top_position,a,1)
    print(top_position)
    left_box=(a, 0, size[0], size[1])  #crop left
    left_position = every_position_crop(left_position, a, 2)

    right_box=(0, 0, size[0]-a, size[1])  #crop right
    right_position = every_position_crop(right_position, a, 3)

    botton_box=(0, 0, size[0], size[1]-a)  #crop botton
    botton_position = every_position_crop(botton_position, a, 4)
    img1 = Image.open(img_p)
    img2 = Image.open(img_p)
    img3 = Image.open(img_p)
    img4 = Image.open(img_p)
    roi1 = img1.crop(top_box)
    roi2 = img2.crop(left_box)
    roi3 = img3.crop(right_box)
    roi4 = img4.crop(botton_box)
    roi1.save('./crop_top.jpg')     #save image
    roi2.save('./crop_left.jpg')
    roi3.save('./crop_right.jpg')
    roi4.save('./crop_botton.jpg')
    trans_function(top_position,"crop_top.jpg")   #save xml
    trans_function(left_position,"crop_left.jpg")
    trans_function(right_position,"crop_right.jpg")
    trans_function(botton_position,"crop_botton.jpg")

def every_position_crop(position,a,flag):
    # position is list of all position
    if flag == 1:
        for i in position:
            position[i][1] = position[i][1] - a
        return position
    elif flag == 2:
        for i in position:
            position[i][0] = position[i][0] - a
        return position
    elif flag == 3:
        for i in position:
            position[i][2] = position[i][2] - a
        return position
    else:
        for i in position:
            position[i][3] = position[i][3] - a
        return position

get_image_and_xml('0.bmp','0.xml')


