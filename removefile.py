import os
import shutil
dir_path = "/home/westwell/hj/hj_data/171129_pg_addR_addPF_blur/char/data/JPEGImages"
save_path = "/home/westwell/hj/hj_data/collect/all_1/char/data/JPEGImages"
a = 1
for file in os.listdir(dir_path):
	shutil.copy(os.path.join(dir_path,file),os.path.join(save_path,file))
	a = a+1
	print a
print "yes"
print a