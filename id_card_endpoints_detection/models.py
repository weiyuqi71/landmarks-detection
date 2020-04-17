## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self,n_points):
        super(Net, self).__init__()
        self.n_points = n_points
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.3)
        self.avg_pool = nn.AdaptiveAvgPool2d(7)
        self.fc1 = nn.Linear(256 * 7 * 7, 2048)
        self.bn1f = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2f = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3f = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, self.n_points*2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.relu(self.bn1(self.conv1(x)))
        # output = (224 - 4 + 2)/2 + 1 = 112 | 32 * 112 * 112
        x = self.relu(self.bn2(self.conv2(x)))
        # output = (112 - 4 + 2)/2 + 1 = 56 | 64 * 56 * 56
        x = self.drop(x)
        x = self.relu(self.bn3(self.conv3(x)))
        # output = (56 - 4 + 2)/2 + 1 = 28 | 128 * 28 * 28 
        x = self.drop(x)
        x = self.relu(self.bn4(self.conv4(x)))

        # output = (28 - 3 + 2)/1 + 1 = 28 | 256 * 28 * 28
        x = self.avg_pool(x) 
        # output = 7 | 256 * 7 * 7
        x = x.reshape(x.shape[0], -1)

        x = self.relu(self.bn1f(self.fc1(x)))
        x = self.relu(self.bn2f(self.fc2(x)))
        x = self.relu(self.bn3f(self.fc3(x)))
        x = self.fc4(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
