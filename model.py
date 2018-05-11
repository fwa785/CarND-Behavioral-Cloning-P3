import argparse
import csv
import cv2
import numpy as np
import sklearn
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.layers.core import Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from sklearn.model_selection import train_test_split

# The function to return the shape of the training data image
def find_image_shape(dirname, lines):
    source_path = lines[0][0]
    filename = source_path.split('\\')[-1]
    current_path = dirname + '/IMG/' + filename
    image = cv2.imread(current_path)
    return image.shape

# The function to load the training/validation image locations from CSV file
def load_csv_lines_from_dir(dirname):
    lines = []
    with open(dirname+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    train_lines, validation_lines = train_test_split(lines, test_size=0.2)

    image_shape = find_image_shape(dirname, train_lines)

    return train_lines, validation_lines, image_shape

# The generator function to allow process the big data set in batches
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
                for i in range(3): # Load center, left, right images
                    source_path = line[i]
                    filename = source_path.split('\\')[-1]
                    current_path = dirname + '/IMG/' + filename
                    image = cv2.imread(current_path)
                    # convert the color from BGR to RGB because drive.py
                    # uses RGB color space and imread uses BGR space
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                measurement = float(line[3])
                # center image
                measurements.append(measurement)
                # left image add turn right steering correction
                measurements.append(measurement + correction)
                # right image add turn left steering correction
                measurements.append(measurement - 2 * correction)

            # augmented the data by flipping the images
            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                flipped_image = cv2.flip(image, 1)
                flipped_measurement = measurement * -1.0
                augmented_images.append(flipped_image)
                augmented_measurements.append(flipped_measurement)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# The function defines the training model, train the model and save the trained model
def train_model(dirname, model_filename):
    #load data from data directory
    train_lines, validation_lines, input_shape = load_csv_lines_from_dir(dirname)

    # compile and train the model using the generator function
    train_generator = generator(dirname, train_lines, batch_size=32)
    validation_generator = generator(dirname, validation_lines, batch_size=32)

    model = Sequential()
    #Normalize the image
    model.add(Lambda(lambda x: (x - 128.)/128., input_shape=input_shape))
    #Crop the top and bottom part of the image because they're not useful for decision making
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    # The following implemented the Nvidia Autonomous Driving Pipeline
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # Use Adam optimizer and mean squared error for loss
    model.compile(optimizer='adam', loss='mse')
    model.fit_generator(train_generator,samples_per_epoch=len(train_lines) * 6,
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_lines) * 6, nb_epoch=7)

    model.save(model_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Driving')

    # The first parameter for the directory to load the training data
    parser.add_argument(
        'data_dir',
        type=str,
        help='The directory to load the data'
    )

    # The second parameter for filename to save the trained model
    parser.add_argument(
        'model_name',
        type=str,
        help='The saved filename of the model'
    )

    args = parser.parse_args()

    train_model(args.data_dir, args.model_name)