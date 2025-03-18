import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .simam import SimAMModule


class SegmentationNet(nn.Module):
    """
    图像分割模型：基于卷积神经网络的分割模型。
    """

    def __init__(self, input_dim, num_channels, num_convs):
        super(SegmentationNet, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(input_dim, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # 中间卷积层
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.simam_layers = nn.ModuleList()
        for _ in range(num_convs - 1):
            self.conv_layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(num_channels))
            self.simam_layers.append(SimAMModule(num_channels))

        # 最后一层卷积
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_channels)

        # 残差连接
        self.residual = nn.Conv2d(input_dim, num_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = self.residual(x)
        # 第一层卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # 中间卷积层
        for conv, bn, simam in zip(self.conv_layers, self.bn_layers, self.simam_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = simam(x)    #

        # 最后一层卷积
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + residual  # 残差连接
        return x