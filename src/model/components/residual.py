import torch
from torch import nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    The building blocks of ResNets
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels or stride != 1:
            self.conv3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResidualBlockTranspose(nn.Module):
    """
    A transpose version of ResidualBlock
    """

    def __init__(self, in_channels, out_channels, stride=1, output_padding=0):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            output_padding=output_padding,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels or stride != 1:
            self.conv3 = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                output_padding=output_padding,
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
