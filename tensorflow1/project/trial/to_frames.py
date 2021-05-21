# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:43:46 2019

@author: Sir
"""

import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('inp_vid.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("frames/frame%d.png" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
  