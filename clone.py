import csv
import cv2
import numpy as np
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.layers.core import Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D

# read in the data csv file
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#correction
correction = 0.3

#load the images
images = []
measurements = []
for line in lines[1:]:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
    measurement = float(line[3])
    #center image
    measurements.append(measurement)
    #left image
    measurements.append(measurement + correction)
    #right image
    measurements.append(measurement - correction)

#augment the data with image flip
augmented_images = []
augmented_measurements = []
for image, measurementy in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print(X_train.shape, y_train.shape)

input_shape = X_train.shape[1:]

model = Sequential()
model.add(Lambda(lambda x: (x - 128.)/128., input_shape=input_shape))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(84, activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')