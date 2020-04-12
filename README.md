[image1]: ./img/facial_keypoints_example.PNG "Facial Keypoint Detection"
[image2]: ./img/facial_keypoints_dataset.PNG "Facial Keypoints Dataset"
[image3]: ./img/model_architecture.PNG "Model Architecture"
[image4]: ./img/loss_result.PNG "Loss Result"
[image5]: ./img/Haar_result.PNG "Haar Result"

# Facial Keypoint Detection
Detecting 68 facial keypoints that include points around eyes, nose, and mouth on a face. 

![Facial Keypoint Detection][image1]

## Overview

1. Defining CNN architecture and training test data
2. Detect faces in an image using a face detector (Haar Cascade)
3. Detect facial keypoint


## Data
This set of image data has been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos.

#### Training and Testing Data

This facial keypoints dataset consists of 5770 color images. All of these images are separated into either a training or a test set of data.

* 3462 of these images are training images, for you to use as you create a model to predict keypoints.
* 2308 are test images, which will be used to test the accuracy of your model.

The information about the images and keypoints in this dataset are summarized in CSV files, which we can read in using `pandas`. Let's read the training CSV and get the annotations in an (N, 2) array where N is the number of keypoints and 2 is the dimension of the keypoint coordinates (x, y).

In each training and test image, there is a single face and **68 keypoints, with coordinates (x, y), for that face**.  These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc.

![Facial Keypoints Dataset][image2]


## Defining CNN architecture and training train data

#### Model Architecture

The model architecture design based on VGGNet A-model (VGG11). Unlike the VGG11, gray scale image (224x224) is input instead of using color image (224x224).

![Model Architecture][image3]

#### Data loading and create the transformed dataset
A sample of dataset will be a dictionary {'image': image, 'keypoints': key_pts}. 

```python
# dataset class
class FacialKeypointDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = self.root_dir + self.key_pts_frame.iloc[idx, 0]
        image = mpimg.imread(image_name)

        if (image.shape[2] == 4):
            image = image[:,:,0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample


# define data transform
data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])

# create the transformed dataset
face_dataset = FacialKeypointDataset(csv_file="../data/training_frames_keypoints.csv", 
                                     root_dir="../data/training/", 
                                     transform=data_transform)
```



#### Hyperparameters



#### Training train data

![Loss Result][image4]



## Detect faces in an image using a face detector (Haar Cascade)

```python
img = cv2.imread('img/ys.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img, 1.2, 2)

img_with_detections = img.copy()

for (x,y,w,h) in faces:
    cv2.rectangle(img_with_detections, (x,y), (x+w,y+h), (255,0,0),3)
```

![Haar Result][image5]


## Detect facial keypoint




