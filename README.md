[image1]: ./img/facial_keypoints_example.PNG "Facial Keypoint Detection"
[image2]: ./img/facial_keypoints_dataset.PNG "Facial Keypoints Dataset"
[image3]: ./img/model_architecture.PNG "Model Architecture"
[image4]: ./img/loss_result.PNG "Loss Result"
[image5]: ./img/Haar_result.PNG "Haar Result"
[image6]: ./img/facial_keypoints_result.PNG "Facial Keypoints Result"

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

```python
batch_size = 32
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.999), eps=1e-08)
n_epochs = 300;
```


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

#### Training data
```python
def train_net(n_epochs):
    # prepare the net for training
    model.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']
            
            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # transfer to GPU
            images, key_pts = images.to(device), key_pts.to(device)

            # forward pass to get outputs
            output_pts = model(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Training')
```

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


## Detect facial keypoints by using trained CNN


```python
net = Net()
print(net)

# load saved trained network
net.load_state_dict(torch.load('saved_models/keypoints_model_5_batch64_epoch60.pt'))
net.eval()

image_copy = np.copy(img)

for (x,y,w,h) in faces:
    height, width = image_copy.shape[:2]

    h_start = int(max(y-h/2, 0))
    h_end = int(min(y+h+h/2, height))
    w_start = int(max(x-w/2, 0))
    w_end = int(min(x+w+w/2, width))

    roi = image_copy[h_start:h_end, w_start:w_end]
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    roi = roi / 255.0

    output_size = 224
    h, w = roi.shape[:2]

    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size
    
    new_h, new_w = int(new_h), int(new_w)
    
    roi = cv2.resize(roi, (new_w, new_h))

    if(len(roi.shape) == 2):
        roi = roi.reshape(roi.shape[0], roi.shape[1], 1)
            
    roi = roi.transpose((2, 0, 1))
    roi = torch.from_numpy(roi)

    roi = roi.type(torch.FloatTensor).unsqueeze(0)
    key_pts = net(roi)
    
    roi = roi.squeeze()

    key_pts = key_pts.view(68, -1)
    key_pts = key_pts.data.numpy() * 50.0 + 100

    plt.figure()
    roi = image_copy[h_start:h_end, w_start:w_end]
    roi = cv2.resize(roi, (new_w, new_h))

    plt.imshow(roi)
    plt.scatter(key_pts[:,0], key_pts[:,1], s=20, marker='.', c='m')


```

![Facial Keypoints Result][image6]

