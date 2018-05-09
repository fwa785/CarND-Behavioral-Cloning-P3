import csv
import cv2
import numpy as np
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.layers.core import Lambda

# read in the data csv file
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#load the images
images = []
measurements = []
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape, y_train.shape)

input_shape = X_train.shape[1:]

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
model.add(Flatten())
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')