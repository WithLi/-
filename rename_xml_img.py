import os

base_dir = "/home/westwell/hj/qz"
img_dir = os.path.join(base_dir, 'data/JPEGImages')
ano_dir = os.path.join(base_dir, 'data/Annotations')


a = 150000
#delete    the  picture of the no xml file pic
for imagename in os.listdir(img_dir):
    img_fpath = os.path.join(img_dir, imagename)
    ano_fpath = os.path.join(ano_dir, imagename[:-4]+'.xml')
    print(os.path.join(img_dir,str(a)+imagename[-4:]))
    print(os.path.join(ano_dir,str(a)+'.xml'))
    os.rename(img_fpath,os.path.join(img_dir,str(a)+imagename[-4:]))
    os.rename(ano_fpath,os.path.join(ano_dir,str(a)+'.xml'))
    a = a+1