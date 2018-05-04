# -*- coding: utf-8 -*-
import cv2,os,shutil
# src_dir = '/home/wy/Downloads/PSA视频错误分析/video1/cam1/0/清晰识别错'
src_dir = '/home/westwell/wy/test_video/zhakou/cv/test_videos/qinzhou/video/j3/test'
dst_dir = '/home/westwell/wy/test_video/zhakou/cv/test_videos/qinzhou/video/j3/save_test'
# video_names = ['TCLU4424788.avi']
video_names = os.listdir(src_dir)
for video_name in video_names:
    video_fpath = os.path.join(src_dir,video_name)
    print video_fpath
    vidcap = cv2.VideoCapture(video_fpath)
    if os.path.exists(os.path.join(dst_dir, video_name[:-4])):
        shutil.rmtree(os.path.join(dst_dir, video_name[:-4]))
    os.makedirs(os.path.join(dst_dir, video_name[:-4]))
    # vidcap.set(cv2.CAP_PROP_POS_MSEC,2000)      # just cue to 20 sec. position
    success, image = vidcap.read()
    count = 0
    print success
    while success:
        success, image = vidcap.read()
        if count%1 == 0:
            cv2.imwrite(
            os.path.join(dst_dir, video_name[:-4],'frame%d.jpg '% count),
            image)  # save frame as JPEG file
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break
        count += 1
