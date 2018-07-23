import numpy as np
import tensorflow as tf
import os
import cv2
import xml.etree.ElementTree as ET
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
dict_elect = []
dict_map = {}
class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = r'/headless/hj/Docker/ssd_model_container/type_new.pb'
        self.PATH_TO_LABELS = r'/headless/hj/Docker/models-master/tf_test/container_class_label/coniner_type_map.pbtxt'
        self.NUM_CLASSES = 36
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def readAnoXML(self, filefpath_xml):
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

    def bbox_overlap(self, g_bbox, p_bbox):
        area1 = (g_bbox[3] - g_bbox[1]) * (g_bbox[2] - g_bbox[0])
        area2 = (p_bbox[3] - p_bbox[1]) * (p_bbox[2] - p_bbox[0])
        # calculate overlap section's coordinate(top left and bottom right)
        xx1 = np.maximum(g_bbox[0], p_bbox[0])
        yy1 = np.maximum(g_bbox[1], p_bbox[1])
        xx2 = np.minimum(g_bbox[2], p_bbox[2])
        yy2 = np.minimum(g_bbox[3], p_bbox[3])
        # width
        w = np.maximum(0.0, xx2 - xx1 + 1)
        # height
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        overlap = inter * 1.0 / (area1 + area2 - inter)
        return overlap

    def eval_bbox(self, charlist, charcoord, pred_result):
        #print(charlist)
        #print(charcoord
        correct_classes = []
        OVERLAP_THRESH = 0.5
        for i in range(len(charlist)):
            char = charlist[i]
            g_bbox = charcoord[i]
            if char in pred_result:
                dets = pred_result[char]
                overarea = self.bbox_overlap(g_bbox, dets)
                if overarea > OVERLAP_THRESH:
                    correct_classes.append(char)
        return correct_classes
    def self_map_to_test_map(self):    #get test map {1: u'wordh', 2: u'numhl', 3: u'numhs', 4: u'wordv', 5: u'numvl', 6: u'numvs'}

        test = self._load_label_map()
        re_dict = {}
          #template save objecte and  index
        for key in test:
            tem_dict = []
            #print(test.get(key))
            for key2 in test.get(key):
                tem_dict.append(test.get(key).get(key2))
            re_dict[tem_dict[0]] = tem_dict[1]
        return re_dict

    def detect(self,sess, image,lis,pos):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        char_class = []
        char_boxes = []
        char_map = {}
        char_dict = self.self_map_to_test_map()
        #print len(boxes[0]),len(scores[0]),len(classes[0])
        for cl in range(len(classes[0])):
            if(scores[0][cl] > 0.5):
                char_class.append(classes[0][cl])
                #print(int(image.shape[0]),int(image.shape[1]))
                new_x1 = int(boxes[0][cl][1] * int(image.shape[1]))
                new_y1 = int(boxes[0][cl][0] * int(image.shape[0]))
                new_x2 = int(boxes[0][cl][3] * int(image.shape[1]))
                new_y2 = int(boxes[0][cl][2]* int(image.shape[0]))
                char_boxes.append(boxes[0][cl])
                char_map_key = char_dict.get(int(classes[0][cl]))
                char_map[char_map_key] = [new_x1,new_y1,new_x2,new_y2]
       # print(char_boxes)
        correct_classes=self.eval_bbox(lis,pos,char_map)
        return correct_classes

if __name__ == '__main__':

    detecotr = TOD()

    data_path = "/headless/hj/hj_data/container_all_cg/20171219/container/type/"
    xml_path = data_path + "data/"+"Annotations"
    img_path = data_path + "data/"+"JPEGImages"
    total_dict ={}
    predict_dict = {}
    total_num = 0
    right_num = 0

    with detecotr.detection_graph.as_default():
        sess = tf.Session(graph=detecotr.detection_graph)
        for file1 in os.listdir(img_path):
            print(file1)
            image = cv2.imread(os.path.join(img_path,file1))
            charlist, charcoord = detecotr.readAnoXML(os.path.join(xml_path,file1[:-4]+'.xml'))
            pre_class = detecotr.detect(sess,image,charlist,charcoord)
            for i in range(len(charlist)):
                total_dict[charlist[i]] = total_dict.get(charlist[i],0) + 1
            if pre_class is not None:
                for i in range(len(pre_class)):
                    predict_dict[pre_class[i]] = predict_dict.get(pre_class[i],0) + 1
                if len(charlist) == len(pre_class):
                    right_num = right_num + 1
            total_num = total_num + 1
        total_char = 0
        corr_char = 0
        for one_char in total_dict:
            total_char = total_char + total_dict[one_char]
            corr_char = corr_char + predict_dict.get(one_char,0)
            print(one_char + "  " + str(total_dict[one_char]) + "  " + str(predict_dict.get(one_char,0)))
        print("total_char :  " + str(total_char) + " correct : " + str(corr_char))
        print("total :  "+str(total_num)+"  correct :"+str(right_num))
        print right_num * 1.0/total_num

