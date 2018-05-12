# **Behavioral Cloning** 

## Overview

---

In this project, I designed a convolution neural network to clone the driving behavior from human. 

It's done in the following steps:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one and track two without leaving the road

[//]: # (Image References)

[image1]: ./examples/center_2018_05_10_11_32_25_173.jpg "Track 1 Centeral Image"
[image2]: ./examples/center_2018_05_10_12_54_25_203.jpg "Track 2 Centeral Image"
[image3]: ./examples/center_2018_05_10_12_54_25_203_flipped.jpg "Track 2 Flipped Image"


---
## Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* track1_model.h5 containing a trained convolution neural network for track one
* track2_model.h5 containing a trained convolution neural network for track two
* writeup.md to summarize the results
* track1_run.mp4 for the video clips for autonomous driving on track one
* track2_run.mp4 for the video clips for autonomous driving on track two

### 1. Functional Code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track one by executing 
```sh
python drive.py track1_model.h5
```
and around the track two by executing
```sh
python drive.py track2_model.h5
```
To make the autonomous driving for track 2 works better, I changed the speed in drive.py from 9 to 12. 

### 2. Usable and Readable Code

The model.py takes two parameters: first parameter for the directory of training data, the second parameter for the file to save the trained model. It loads the training data from the given directory, split the data into training and validation, training the network with the training data and save the trained network to the filename given by the command line. 

An example to run the model:
```sh
python model.py track1_data track1_model.h5
```

The model.py has the comments in the code to help explain the design.

## Model Architecture and Training Strategy

### 1. Model Architecture

My model is pretty much the replicate of the Nvidia's self drive car pipeline: 


| Layer         		|     Description	        										| 
|:---------------------:|:-----------------------------------------------------------------:| 
| Input         		| 160x320x3 color image   											| 
| Normalization   		| Normalize the image by pixel-128/128, outputs 160x320x3			| 
| Cropping   			| Cropping the top 70 pixels and bottom 25 pixels, outputs 65x320x3	| 
| Convolutional 5x5 	| 2x2 stride, same padding, outputs 31x158x24 						|
| RELU					|																	|
| Convolutional 5x5 	| 2x2 stride, same padding, outputs 14x77x36 						|
| RELU					|																	|
| Convolutional 5x5 	| 2x2 stride, same padding, outputs 5x37x48 						|
| RELU					|																	|
| Convolutional 3x3 	| 1x1 stride, same padding, outputs 3x35x64 						|
| RELU					|																	|
| Convolutional 3x3 	| 1x1 stride, same padding, outputs 1x33x64 						|
| RELU					|																	|
| Flatten				| outputs 2112														|
| RELU					|																	|
| Fully Connected		| output 100														|
| RELU					|																	|
| Fully Connected		| output 50															|
| RELU					|																	|
| Fully Connected		| output 10															|
| RELU					|																	|
| Fully Connected		| output 1													
|
|						|																	|
|						|																	|

The data is first normalized in the model using Keras lamada layer(code line 95). Then it uses Keras Cropping2D layer to trim the top and bottom part of the image to avoid unnecessary information(code line 97). The it uses 3 5x5 kernel size convolutional layers with RELU activation(code line 99-101), followed by 2 3x3 kernel size convolutional layers with RELU activation(code line 102-103), followed by 4 fully connected layers(code line 104-108).

### 2. Attempts to reduce overfitting in the model

The data was split into training and validation data by 80/20 ratio. The model was trained on training data and validated by validation data to avoid overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I didn't use dropout layers to reduce the overfitting. Surprisely, when I used the dropout layer, it trained the model worse, so I had to remove them.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

### 4. Appropriate training data

I tried to drive the car carefully to keep the car on the center when I collected the data. For track 2, I really couldn't keep the car on the right lane only, but I still kept the car stay on the track. I didn't collect the data to recover from left or right seperately. However, since I didn't always drive the car perfectly, especially for track 2, there are probably lots of data for recovering from left and right. For track 1, I collected data for about 2 laps, and for track 2, I probably collected the data for 4-5 laps.

### 5. Solution Design Approach

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I started with LeNet convolutional neural network because it's simple and easy to implement. I only loaded the images from the central camera. This model has the training loss repidly going down but the validation loss stays the same through the epoches. After 7 epoches, the validation loss is signficantly higher than the training loss. That implies the model was overfitting. 

Then I did the data augmentation by adding the images flipped horizontally with steering angle at opposite direction. This seemed improved the overfitting. The gap between the training loss and the validation loss is smaller, although the validation loss is still significantly higher than the training loss. This means the model was still overfitting.

Next I used the images from left camera and right camera and adjusted the steering angles accordingly: the left camera images adjust the steering angles to the right, and the right camera images adjust the steering angles to the left. With the additional data, the gap between the traing loss and the validation loss is even smaller. But both of the loss are higher than before. Even adding number of epoches doesn't help. This means we need to improve the neural network.

I tried to run autonomous driving with the saved model file, it didn't go through 1/4 laps.

However, what really made the difference for the autonomous driving is the BGR2RBG convertion. Without the BGR2RBG conversion, the car can barely drive for more than 1/4 lap. But after I found out that the image read by imread is in BGR color space and the drive.py uses RBG color space, and after did the color conversion, the driving improved dramatically. It could finish drive in one lap without off the track.

Later on, I found out with BGR2RBG color conversion, even with the data before augmentation, the model can be trained to drive one lap. It's amazing!!

But the track 2 is more challenging. The model can't be trained to train on track 2 for one lap. Therefore, I changed the LeNet neural network to Nvidia's self drive car pipeline. After that, it still couldn't drive well. Then I collected more data for track 2 by driving at least 4-5 laps. With more data, I had to implement the data generator to avoid memory error. With more data and Nvidia pipeline, the car drove much better but at some uphill it stopped. I then changed the speed in drive.py from 9 to 12, and autonomous drive work for track 2 as well. 

At the end of the process, the vehicle is able to drive autonomously around both the track 1 and track 2 without leaving the road.

### 2. Final Model Architecture

The final model architecture (model.py lines 93-108) consisted of a convolution neural network with 5 convolutional layers and 4 fully connected layers.

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded about two laps on track one using center lane driving. Here is an example image of center lane driving for track 1:

![alt text][image1]

I recorded about four to five laps on track two using center road driving. I tried to stay at the right lane, but it is really hard for me to control it with the keyboard. Here is an example image of center road driving for track 2:

![alt text][image2]

I didn't record any data to recover the vehicle from left side or right side of the road. 

After the collection process, I had 4152 samples for track 1 and 8778 samples for track 2. Each sample includes 3 images: center, left, and right image. 

To augment the data sat, I flipped images and angles thinking that this would balance the data set with left turn and right turn. Here is original image and the flipped image for a track 2 image. 

![alt text][image2]
![alt text][image3]

Actually because I augmented the data by flipping the image, it makes it's hard to automate driving on the right lane only because the flipped image changes from right lane to the left lane. So if we indeed want to automate driving on the right lane only, we can't use flipping to augment the data.

With augmented data, I have 24912 images for track 1 and 52668 images for track 2. The data is randomly shuffle and split into 80% training and 20% validation data. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7. After 7 epoches, the validation loss starts to increase. I used an adam optimizer so that manually training the learning rate wasn't necessary.
