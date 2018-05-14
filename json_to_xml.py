import xml.dom.minidom
import cv2
import os
import json
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
#def trans_function(json,image_url):

    #bboxes=gts.readlines()
    #print(bboxes)
    #gts.close()
data = '{"n": [28, 11, 51, 55], "gx": [5, 11, 27, 55], "five": [86, 11, 107, 55], "nine": [133, 11, 154, 55], "eight": [63, 12, 83, 55], "three": [156, 11, 178, 56]}'
dict_txt = json.loads(data)
#print(data)
data_dict = eval(data)
#print(type(eval(data)))
object_position = []
object_name = []
image_url = ""
for bbox in data_dict:
    object_name.append(bbox)
    filename = bbox
    #print(type(x_min))
    xmin=int(data_dict[bbox][0])
    ymin=int(data_dict[bbox][1])
    xmax=int(data_dict[bbox][2])
    ymax=int(data_dict[bbox][3])
    #print([xmin,ymin,xmax,ymax])
    object_position.append([xmin,ymin,xmax,ymax])
print(object_name)
print(object_position)
#image_path2 = image_path + xml_name[:-4]+'.jpg'
    #print(image_path2)
    #if(cv2.imread(image_path2) is None):
     #   print(image_path2)
    #else:
     #  img = cv2.imread(image_path2)
width = 1920
height = 1080
depth = 3
xml_filename='./test.xml'
pascal_voc_xml_trans(xml_filename,'armor',filename,'armor','armor','armor',str(width),str(height),str(depth),'0',object_name,'Unspecified','0','0',object_position)

"""
xml_path = "/home/xj/competition/test/txt_9000"
image_path = "/home/xj/competition/test/image/"
for xml1 in os.listdir(xml_path):
    trans_function(xml_path,image_path,xml1)"""