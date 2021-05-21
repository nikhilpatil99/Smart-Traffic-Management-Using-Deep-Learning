# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:07:04 2019

@author: Sir
"""

from imageai.Detection import ObjectDetection
import os
'''
execution_path = os.getcwd()

detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "D:/Project/obj/training_set/models/models.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(trucks=False, cars=True)
detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , "car.png"), output_image_path=os.path.join(execution_path , "image_new.png"), custom_objects=custom_objects, minimum_percentage_probability=65)


for eachObject in detections:
   print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
   print("--------------------------------")
'''
import h5py

filename = "./models.h5"

h5 = h5py.File(filename,'r')

futures_data = h5['futures_data']  # VSTOXX futures data
options_data = h5['options_data']  # VSTOXX call option data

h5.close()