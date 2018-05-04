import os
import shutil
xml_path = "/home/westwell/hj/hj_data/collect/no_factory_char/char/data/Annotations"
img_path = "/home/westwell/hj/hj_data/collect/no_factory_char/char/data/JPEGImages"
xml_save = "/home/westwell/hj/hj_data/collect/no_factory_char/test_sub/data/Annotations"
img_save = "/home/westwell/hj/hj_data/collect/no_factory_char/test_sub/data/JPEGImages"

def getsome_pic(xml_path,img_path):
	a = 0
	for file in os.listdir(xml_path):
		shutil.copy(os.path.join(xml_path,file),os.path.join(xml_save,file))
		shutil.copy(os.path.join(img_path,file[:-4]+'.bmp'),os.path.join(img_save,file[:-4]+'.bmp'))
		if a == 10000:
			print "over"
			break;
		a = a+1
	print a


getsome_pic(xml_path,img_path)