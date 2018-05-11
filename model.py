import csv
import cv2
import numpy as np
import sklearn
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.layers.core import Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from sklearn.model_selection import train_test_split

def find_image_shape(dirname, lines):
    source_path = lines[0][0]
    filename = source_path.split('\\')[-1]
    current_path = dirname + '/IMG/' + filename
    image = cv2.imread(current_path)
    return image.shape

def load_csv_lines_from_dir(dirname):
    lines = []
    with open(dirname+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    train_lines, validation_lines = train_test_split(lines, test_size=0.2)

    image_shape = find_image_shape(dirname, train_lines)

    return train_lines, validation_lines, image_shape

def generator(dirname, lines, batch_size=32):
    # correction
    correction = 0.2

    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_lines:
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('\\')[-1]
                    current_path = dirname + '/IMG/' + filename
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                measurement = float(line[3])
                # center image
                measurements.append(measurement)
                # left image
                measurements.append(measurement + correction)
                # right image
                measurements.append(measurement - 2 * correction)

            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                flipped_image = cv2.flip(image, 1)
                flipped_measurement = measurement * -1.0
                augmented_images.append(flipped_image)
                augmented_measurements.append(flipped_measurement)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

#load data from data directory
dirname = 'track2_data2'
train_lines, validation_lines, input_shape = load_csv_lines_from_dir(dirname)

# compile and train the model using the generator function
train_generator = generator(dirname, train_lines, batch_size=32)
validation_generator = generator(dirname, validation_lines, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: (x - 128.)/128., input_shape=input_shape))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator,samples_per_epoch=len(train_lines),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_lines), nb_epoch=7)

model.save('model.h5')