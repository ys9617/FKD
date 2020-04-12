import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1_32 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv32_64 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv64_128 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv128_128 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv128_256 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv256_256 = nn.Conv2d(256, 256, 3, padding=1)

        self.fc1 = nn.Linear(256 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 68 * 2)

        self.drop = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2,2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1_32(x)))     # (32, 112, 112)
        x = self.pool(F.relu(self.conv32_64(x)))   # (64, 56, 56)
        x = F.relu(self.conv64_128(x))             # (128, 56, 56)
        x = self.pool(F.relu(self.conv128_128(x)))  # (128, 28, 28)
        x = F.relu(self.conv128_256(x))             # (256, 28, 28)
        x = self.pool(F.relu(self.conv256_256(x)))  # (256, 14, 14)
        x = F.relu(self.conv256_256(x))             # (256, 14, 14)
        x = self.pool(F.relu(self.conv256_256(x)))  # (256, 7, 7)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
 
        return x
