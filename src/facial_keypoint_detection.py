import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
from models import Net

img = cv2.imread('../img/ys.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

face_cascade = cv2.CascadeClassifier('../detector_architectures/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img, 1.2, 2)

img_with_detections = img.copy()

for (x,y,w,h) in faces:
    cv2.rectangle(img_with_detections, (x,y), (x+w,y+h), (255,0,0),3)


net = Net()
print(net)

net.load_state_dict(torch.load('../saved_models/facial_keypoint_detection.pt'))
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

    roi = image_copy[h_start:h_end, w_start:w_end]
    roi = cv2.resize(roi, (new_w, new_h))
    

plt.figure()

# input image
plt.subplot(1, 3, 1)
plt.imshow(img)

# face detection result
plt.subplot(1, 3, 2)
plt.imshow(img_with_detections)

# facial keypoints result
plt.subplot(1, 3, 3)
plt.imshow(roi)
plt.scatter(key_pts[:,0], key_pts[:,1], s=20, marker='.', c='m')

plt.show()





