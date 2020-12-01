# flake8: noqa

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=0, bias=0):
        super(Conv2dBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)
        return outputs


class FaceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_bn_1 = Conv2dBatchNorm(2, 64, kernel_size=4, stride=1)
        self.conv_bn_2 = Conv2dBatchNorm(64, 64, kernel_size=4, stride=1)
        self.conv_bn_3 = Conv2dBatchNorm(64, 128, kernel_size=4, stride=1)
        # stride equals kernel_size
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv_bn_4 = Conv2dBatchNorm(128, 128, kernel_size=4, stride=1)
        # stride equals kernel_size
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv_bn_5 = Conv2dBatchNorm(128, 128, kernel_size=4, stride=1)
        # stride equals kernel_size
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv_bn_6 = Conv2dBatchNorm(128, 256, kernel_size=4, stride=1)
        # stride equals kernel_size
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv_bn_7 = Conv2dBatchNorm(256, 512, kernel_size=4, stride=1)
        self.conv_bn_8 = Conv2dBatchNorm(512, 512, kernel_size=4, stride=2)
        self.conv_9 = nn.Conv2d(512, 512, kernel_size=4, stride=2)
        self.avgpool_1 = nn.AvgPool2d(kernel_size=(6, 1), stride=1)
        self.bn_9 = nn.BatchNorm2d(512)
        self.relu_9 = nn.ReLU(inplace=True)
        self.fc_1 = nn.Linear(512, 4096)
        self.relu_10 = nn.ReLU(inplace=True)
        self.fc_2 = nn.Linear(4096, 4096)

    def forward(self, x):
        x = self.conv_bn_1(x)
        x = self.conv_bn_2(x)
        x = self.conv_bn_3(x)
        x = self.maxpool_1(x)
        x = self.conv_bn_4(x)
        x = self.maxpool_2(x)
        x = self.conv_bn_5(x)
        x = self.maxpool_3(x)
        x = self.conv_bn_6(x)
        x = self.maxpool_4(x)
        x = self.conv_bn_7(x)
        x = self.conv_bn_8(x)
        x = self.conv_9(x)
        x = self.avgpool_1(x)
        x = self.bn_9(x)
        x = self.relu_9(x)
        x = self.fc_1(x)
        x = self.relu_10(x)
        outputs = self.fc_2(x)

        return outputs


if __name__ == "__main__":
    model = FaceEncoder()
