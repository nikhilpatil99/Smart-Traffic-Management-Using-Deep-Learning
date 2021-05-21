# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:18:20 2019

@author: Sir
"""

import cv2
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('frames/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project_video1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 8, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()