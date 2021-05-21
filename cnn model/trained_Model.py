# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:09:22 2018

@author: yatheen!
"""

# Step 1: Import the packages
from keras.models import model_from_json
import cv2
import numpy as np

# Step 2: Load the Model from Json File
json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Step 3: Load the weights
loaded_model.load_weights("./model.h5")
print("Loaded model from disk")

# Step 4: Compile the model
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Step 5: load the image you want to test
#image = cv2.imread('./training_set/cats/cat.6.jpg')
#image = cv2.imread('./training_set/dogs/dog.52.jpg')
str='./test/t5.png'
image = cv2.imread(str)
image1 = cv2.imread(str)
image = cv2.resize(image, (64,64))
image = image.reshape(1, 64, 64,3)

# Step 6: Predict to which class your input image has been classified
result = loaded_model.predict_classes(image)
if(result[0][0] != 1):
    str1='Car'
    print("I guess this must be a car:!",result[0][0])
else:
    str1='Truck'
    print("I guess this must be a truck:!",result[0][0])


cv2.imshow(str1, image1)
cv2.waitKey(0)
cv2.destroyAllWindows()



   