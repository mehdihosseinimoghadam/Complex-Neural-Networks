"""
Complex Valued VGG Net From Scratch
Programmed by Mehdi Hosseini Moghadam

The Base Code Of VGG Has Been Taken From:
   https://github.com/aladdinpersson

*    MIT Licence
*    2022-02-15 Last Update
"""





import torch
import torch.nn as nn  
from complex_neural_net import *

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class Complex_VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(Complex_VGG_net, self).__init__()
        self.in_channels = in_channels
        self.complex_conv_layers = self.complex_conv_layers_block(VGG_types["VGG16"])

        self.complex_linear_block = nn.Sequential(
            CLinear(512 * 7 * 7 , 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            CLinear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            CLinear(4096, num_classes),
        )

    def forward(self, x):
        x = self.complex_conv_layers(x)
        x = x.reshape(x.shape[0], -1,2)
        x = self.complex_linear_block(x)
        return x

    def complex_conv_layers_block(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    CConv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    CBatchnorm(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [CMaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)
