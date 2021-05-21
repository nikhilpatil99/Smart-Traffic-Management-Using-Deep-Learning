#13 Step CNN

#Step 1: Import the required packages

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import plot_model
# Step 2: Initialising the CNN
model = Sequential()
'''
# Step 3: Convolution
model.add(Conv2D(8,  (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


model.add(Conv2D(16, (3, 3), activation = 'relu'))


model.add(Conv2D(32, (3, 3), activation = 'relu'))


model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (8, 8)))

model.add(Conv2D(128,(3, 3), activation = 'relu'))
'''

model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 4: Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 5: Second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 6: Flattening
model.add(Flatten())

# Step 7: Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Step 8: Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Step 9: ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Step 10: Load the training Set
training_set = train_datagen.flow_from_directory('./training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
# Step 11: Classifier Training 
model.fit_generator(training_set,
                         steps_per_epoch = 2000,
                         epochs = 1,
                         validation_steps = 1000)

# Step 12: Convert the Model to json
model_json = model.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

# Step 13: Save the weights in a seperate file
model.save_weights("./model.h5")
plot_model(model, to_file='model.png')
print("Classifier trained Successfully!")