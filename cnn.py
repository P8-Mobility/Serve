import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvNet(nn.Module):
    input_channels = 1
    channels_conv1 = 300
    kernel_conv1 = (5, 30)
    pool_conv1 = [3, 3]
    channels_conv2 = 600
    kernel_conv2 = (2, 10)
    fcl1_size = 50
    output_size = 2

    def __init__(self):
        super(ConvNet, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(self.input_channels, self.channels_conv1, self.kernel_conv1)
        self.conv2 = nn.Conv2d(self.channels_conv1, self.channels_conv2, self.kernel_conv2)

        self.BatchNorm2d = nn.BatchNorm2d(self.channels_conv1)

        # Based on the input vector for the linear layer
        self.conv_out_size = self.input_channels * self.channels_conv2

        # Define the fully connected layers
        self.fcl1 = nn.Linear(self.conv_out_size, self.fcl1_size, bias=False)
        self.fcl2 = nn.Linear(self.fcl1_size, self.output_size, bias=False)

    def forward(self, x: Tensor):
        # Apply convolution 1 and pooling
        x = self.conv1(x)
        x = self.BatchNorm2d(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, self.pool_conv1)

        # Apply convolution 2
        x = self.conv2(x)
        x = F.relu(x)

        # Reshape x to one dimension to use as input for the fully connected layers
        x = x.view(-1, self.conv_out_size)

        # Fully connected layers
        x = self.fcl1(x)
        x = F.relu(x, inplace=True)
        x = self.fcl2(x)

        return x
