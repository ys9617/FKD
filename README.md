[image1]: ./img/key_pts_example.png "Facial Keypoint Detection"
[image2]: ./img/landmarks_numbered.jpg "Facial Keypoint"

# Facial Keypoint Detection
Detecting 68 facial keypoints that include points around eyes, nose, and mouth on a face. 

![Facial Keypoint Detection][image1]

## Overview

1. Defining and training CNN model
2. Detect face position in the image using Haar cascades
3. Detect facial keypoint using trained CNN model


## Data
This set of image data has been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos.

#### Training and Testing Data

This facial keypoints dataset consists of 5770 color images. All of these images are separated into either a training or a test set of data.

* 3462 of these images are training images, for you to use as you create a model to predict keypoints.
* 2308 are test images, which will be used to test the accuracy of your model.

The information about the images and keypoints in this dataset are summarized in CSV files, which we can read in using `pandas`. Let's read the training CSV and get the annotations in an (N, 2) array where N is the number of keypoints and 2 is the dimension of the keypoint coordinates (x, y).

In each training and test image, there is a single face and **68 keypoints, with coordinates (x, y), for that face**.  These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc.

![Facial Keypoint][image2]{: width="30%" height="30%"}

