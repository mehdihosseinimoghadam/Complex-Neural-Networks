"""
Complex Valued Google Inception From Scratch
Programmed by Mehdi Hosseini Moghadam

The Base Code Of Google Inception Has Been Taken From:
   https://github.com/aladdinpersson

*    MIT Licence
*    2022-02-15 Last Update
"""



import torch
from torch import nn
from complex_neural_net import *

class Complex_GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000):
        super(Complex_GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        # Write in_channels, etc, all explicit in self.conv1, rest will write to
        # make everything as compact as possible, kernel_size=3 instead of (3,3)
        self.complex_conv1 = complex_conv_block(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )

        self.complex_maxpool1 = CMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.complex_conv2 = complex_conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.complex_maxpool2 = CMaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.complex_inception3a = complex_Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.complex_inception3b = complex_Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.complex_maxpool3 = CMaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.complex_inception4a = complex_Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.complex_inception4b = complex_Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.complex_inception4c = complex_Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.complex_inception4d = complex_Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.complex_inception4e = complex_Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.complex_maxpool4 = CMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.complex_inception5a = complex_Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.complex_inception5b = complex_Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.complex_avgpool = CAvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.complex_linear = CLinear(1024, num_classes)

        if self.aux_logits:
            self.complex_aux1 = complex_InceptionAux(512, num_classes)
            self.complex_aux2 = complex_InceptionAux(528, num_classes)
        else:
            self.complex_aux1 = self.complex_aux2 = None

    def forward(self, x):
        x = self.complex_conv1(x)
        x = self.complex_maxpool1(x)
        x = self.complex_conv2(x)
        # x = self.conv3(x)
        x = self.complex_maxpool2(x)

        x = self.complex_inception3a(x)
        x = self.complex_inception3b(x)
        x = self.complex_maxpool3(x)

        x = self.complex_inception4a(x)

        # Auxiliary Softmax classifier 1
        if self.aux_logits and self.training:
            complex_aux1 = self.complex_aux1(x)

        x = self.complex_inception4b(x)
        x = self.complex_inception4c(x)
        x = self.complex_inception4d(x)

        # Auxiliary Softmax classifier 2
        if self.aux_logits and self.training:
            complex_aux2 = self.complex_aux2(x)

        x = self.complex_inception4e(x)
        x = self.complex_maxpool4(x)
        x = self.complex_inception5a(x)
        x = self.complex_inception5b(x)
        x = self.complex_avgpool(x)
        x = x.reshape(x.shape[0], -1, 2)
        x = self.dropout(x)
        x = self.complex_linear(x)

        if self.aux_logits and self.training:
            return complex_aux1, complex_aux2, x
        else:
            return x


class complex_Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        super(complex_Inception_block, self).__init__()
        self.complex_branch1 = complex_conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.complex_branch2 = nn.Sequential(
            complex_conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            complex_conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.complex_branch3 = nn.Sequential(
            complex_conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            complex_conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.complex_branch4 = nn.Sequential(
            CMaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            complex_conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

    def forward(self, x):
        return torch.cat(
            [self.complex_branch1(x), self.complex_branch2(x), self.complex_branch3(x), self.complex_branch4(x)], 1
        )


class complex_InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(complex_InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.complex_pool = CAvgPool2d(kernel_size=5, stride=3)
        self.complex_conv = complex_conv_block(in_channels, 128, kernel_size=1)
        self.complex_linear1 = CLinear(2048, 1024)
        self.complex_linear2 = CLinear(1024, num_classes)

    def forward(self, x):
        x = self.complex_pool(x)
        x = self.complex_conv(x)
        x = x.reshape(x.shape[0], -1, 2)
        x = self.relu(self.complex_linear1(x))
        x = self.dropout(x)
        x = self.complex_linear2(x)

        return x


class complex_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(complex_conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.complex_conv = CConv2d(in_channels, out_channels, **kwargs)
        self.complex_batchnorm = CBatchnorm(out_channels)

    def forward(self, x):
        return self.relu(self.complex_batchnorm(self.complex_conv(x)))
