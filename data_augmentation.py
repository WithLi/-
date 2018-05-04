from skimage import exposure,img_as_float
import cv2
import os
import shutil
import re


def chang_light(img_path,save_path,xml_path):
	'''
      # img_path :the image file
	'''
	a = 0
	img_file = os.listdir(img_path)
	print(len(img_file))
	for img in img_file:
		img_url = os.path.join(img_path,img)
		imag = cv2.imread(img_url)
		image = img_as_float(imag)
		xml_p = os.path.join(xml_path,img[:-4] + '.xml')
		gam1 = exposure.adjust_gamma(image,1.5) * 255 # to dark
		save_name1 = save_path + "/" +img[:-4] + "dark.jpg"
		shutil.copy(xml_p,os.path.join(xml_p,os.path.join(xml_path,img[:-4]+'dark.xml')))
		gam2 = exposure.adjust_gamma(image,0.4) * 255
		save_name2 = save_path + "/"+ img[:-4] + "bright.jpg"
		shutil.copy(xml_p,os.path.join(xml_p,os.path.join(xml_path,img[:-4]+'bright.xml')))
		cv2.imwrite(save_name1,gam1)
		cv2.imwrite(save_name2,gam2)
		print(str(a)+"/"+str(len(img_file)))
		a = a+1
chang_light('/home/westwell/hj/hj_data/CG/20171219/container/type/data/some_aug_pic/JPEGImages','/home/westwell/hj/hj_data/CG/20171219/container/type/data/some_aug_pic/JPEGImages','/home/westwell/hj/hj_data/CG/20171219/container/type/data/some_aug_pic/Annotations')

def addnoise(img_path,save_path,xml_path):
	a = 0
	img_file = os.listdir(img_path)
	print(len(img_file))	
	for img in img_file:
		img_url = os.path.join(img_path,img)
		imag = cv2.imread(img_url)
		xml_p = os.path.join(xml_path,img[:-4] + '.xml')
		gam1 = cv2.blur(imag,(3,3)) # to dark
		save_name1 = save_path + "/" +img[:-4] + "noise5.jpg"
		shutil.copy(xml_p,os.path.join(xml_p,os.path.join(xml_path,img[:-4]+'noise5.xml')))
		gam2 = cv2.blur(imag,(1,1))
		save_name2 = save_path + "/"+ img[:-4] + "noise3.jpg"
		shutil.copy(xml_p,os.path.join(xml_p,os.path.join(xml_path,img[:-4]+'noise3.xml')))
		cv2.imwrite(save_name1,gam1)
		cv2.imwrite(save_name2,gam2)
		print(str(a)+"/"+str(len(img_file)))
		a = a+1
addnoise('/home/westwell/hj/hj_data/CG/20171219/container/type/data/some_aug_pic/JPEGImages','/home/westwell/hj/hj_data/CG/20171219/container/type/data/some_aug_pic/JPEGImages','/home/westwell/hj/hj_data/CG/20171219/container/type/data/some_aug_pic/Annotations')





