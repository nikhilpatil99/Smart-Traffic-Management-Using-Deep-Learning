# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:50:37 2019

@author: Sir
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:36:56 2019

@author: Sir
"""

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util 

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'project'
#IMAGE_NAME = '11.jpg'
#IMAGE_NAME = 'truck.png'
PATH_TO_TEST_IMAGES_DIR = 'C:\tensorflow1\project\trial\frames'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{}.png'.format(i)) for i in range(30, 40) ]
# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join('frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join('label_map.pbtxt')

# Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


sys.path
#sys.path.insert(0, 'C:\tensorflow1\project\images\Test')



# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value

for image_path in TEST_IMAGE_PATHS:
  image = cv2.imread(image_path)
  image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
  (boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')
  
  vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.80)
  cv2.imwrite("out_dir/%s" % TEST_IMAGE_PATHS, image)     # save frame as JPEG file    
  break

# All the results have been drawn on image. Now display the image.
  #cv2.imshow('Object detector', image)

# Press any key to close the image
  #cv2.waitKey(0)

# Clean up
  #cv2.destroyAllWindows()