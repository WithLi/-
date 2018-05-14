import xml.dom.minidom
import cv2
import os
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
        text_object_name=dom.createTextNode(name)
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
def trans_function(txt_file,img_path,xml_name):
    result_file = os.path.join(txt_file,xml_name)
    gts=open(result_file,'r')
    bboxes=gts.readlines()
    #print(bboxes)
    gts.close()
    object_position = []
    for bbox in bboxes:
        point = bbox.split(',')
        x = [int(float(point[i])) for i in [0, 2, 4, 6]]
        y = [int(float(point[i])) for i in [1, 3, 5, 7]]
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        bbox=bbox.split()
        filename = bbox[0]
       
        xmin=int(float(x_min))
        ymin=int(float(y_min))
        xmax=int(float(x_max))
        ymax=int(float(y_max))
       
        object_position.append([xmin,ymin,xmax,ymax])
    image_path2 = image_path + xml_name[:-4]+'.jpg'
    
    if(cv2.imread(image_path2) is None):
        print(image_path2)
    else:
        img = cv2.imread(image_path2)
        width,height,depth = img.shape
        xml_filename='PATH'+xml_name[:-4]+'.xml'  #PATH save xml all_path
        pascal_voc_xml_trans(xml_filename,'armor',filename,'armor','armor','armor',str(width),str(height),str(depth),'0','roi','Unspecified','0','0',object_position)



xml_path = "PATH"
image_path = "PATH"
for xml1 in os.listdir(xml_path):
    trans_function(xml_path,image_path,xml1)
