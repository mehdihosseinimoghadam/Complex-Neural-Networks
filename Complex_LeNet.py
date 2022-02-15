"""
Complex Valued LeNet From Scratch
Programmed by Mehdi Hosseini Moghadam

The Base Code Of LeNet Has Been Taken From:
   https://github.com/aladdinpersson

*    MIT Licence
*    2022-02-15 Last Update
"""



import torch
import torch.nn as nn  
from complex_neural_net import *

class Complex_LeNet(nn.Module):
    def __init__(self):
        super(Complex_LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.Complex_pool = CAvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.complex_conv1 = CConv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.complex_conv2 = CConv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.complex_conv3 = CConv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.Complex_linear1 = CLinear(120, 84)
        self.Complex_linear2 = CLinear(84, 10)

    def forward(self, x):
        x = self.relu(self.complex_conv1(x))
        x = self.Complex_pool(x)
        x = self.relu(self.complex_conv2(x))
        x = self.Complex_pool(x)
        x = self.relu(
            self.complex_conv3(x)
        )  # num_examples x 120 x 1 x 1 --> num_examples x 120
        x = x.reshape(x.shape[0], -1, 2)
        x = self.relu(self.Complex_linear1(x))
        x = self.Complex_linear2(x)
        return x
