# Complex Neural Networks 🧠
##### This Repo Contains Implementation of Complex Valued Neural Networks in Pytorch including 🧱:
- Complex Linear Layer
- Complex Convolution2d layer
- Complex ConvolutionTrans2d layer
- Complex BatchNorm2d layer
- Complex MaxPool2d layer
- Complex AvePool2d layer
- Complex LSTM layer
##### And Some Famous Deep Learning Architectures Like 🏛️:

- Complex Valued VGG11, VGG13, VGG16
- Complex Valued LeNet
- Complex Valued Google Inception




[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/Complex_Deep_Neural_Network.ipynb)



#### Short Intro on Complex Neural Networks 📖
Almost all deep learning layers and deep learning models work with real numbers, but there are some cases which we might need complex numbers in our neural net. A brilliant example of this is in the area of signal processing, when we want to analyze both magnitude and phase of a given signal (for more info on that refer to [this paper](https://openreview.net/forum?id=SkeRTsAcYm) So it is important to have complex valued neural networks, which this repo is all about. For more info refer to [this paper](https://arxiv.org/abs/2101.12249)




Prerequisites 🧰
-------------
- `Python 3.6` 
- `pytorch`



Layers 🧱
----------
| Layer  | Class Name |  File | 
| :-------------: | :---------------: | :---------------: |
| Complex Linear Layer | `CLinear` |  [complex_neural_net](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/complex_neural_net.py) |
| Complex Convolution2d layer | `CConv2d` | [complex_neural_net](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/complex_neural_net.py) |
| Complex ConvolutionTrans2d layer | `CConvTrans2d` | [complex_neural_net](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/complex_neural_net.py) |
| Complex BatchNorm2d layer | `CBatchnorm` | [complex_neural_net](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/complex_neural_net.py) |
| Complex MaxPool2d layer | `CMaxPool2d` | [complex_neural_net](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/complex_neural_net.py) |
| Complex AvePool2d layer | `CAvgPool2d` | [complex_neural_net](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/complex_neural_net.py) |
| Complex LSTM layer | `CLSTM` | [complex_neural_net](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/complex_neural_net.py) |




Architectures 🏛️
----------
| Layer  | Class Name |  File | 
| :-------------: | :---------------: | :---------------: |
| Complex Valued VGG11, VGG13, VGG16 | `Complex_VGG_net` |  [Complex_Vgg_net](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/Complex_Vgg_net.py) |
| Complex Valued LeNet | `Complex_LeNet` | [Complex_LeNe](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/Complex_LeNet.py) |
| Complex Valued Google Inception | `Complex_GoogLeNet` | [Complex_Google_Inception](https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/Complex_Google_Inception.py) |





## Usage

Clone the Repo:



```sh
git clone https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks.git
```

Some Imports
```py
>>> import torch
>>> from complex_neural_net import CConv2d
```

Praper Complex Valued Data

```py
>>> x0 = torch.randn(5,5)
>>> x1 = torch.randn(5,5)
>>> x = torch.stack([x0,x1],-1)
>>> x = x.unsqueeze(0)
>>> x = x.unsqueeze(0)
>>> print(x.shape)

torch.Size([1, 1, 5, 5, 2])
```


Use Complex Valued Conv2d

```py
>>> CConv2d1 = CConv2d(in_channels = 1, out_channels = 2, kernel_size = (2,2), stride = (1,1), padding = (0,0))
>>> print(x.shape)
>>> print(CConv2d1(x))
>>> print(CConv2d1(x).shape)

torch.Size([1, 1, 5, 5, 2])
tensor([[[[[-2.7639, -0.3970],
           [-1.5627,  0.3068],
           [ 2.3798,  1.2708],
           [-1.1730,  2.1180]],

          [[ 2.4931,  0.5094],
           [-0.9082, -2.1115],
           [ 1.4688, -2.2492],
           [-0.4631,  1.0015]],

          [[-1.1452,  1.4262],
           [-1.0511,  4.3379],
           [ 0.9986,  0.9051],
           [ 2.2954,  1.1620]],

          [[-0.1294,  0.9085],
           [ 0.5013,  0.3251],
           [-1.1305,  1.0306],
           [ 0.0047,  0.9547]]],


         [[[-2.3747, -0.1068],
           [-2.0242,  0.8044],
           [-0.8330, -1.5812],
           [ 0.1164,  0.0097]],

          [[ 1.6645, -0.8150],
           [ 0.0091, -0.3579],
           [-1.6963, -2.1597],
           [ 0.5094, -0.8979]],

          [[-1.1619, -0.5089],
           [ 0.4402,  1.2927],
           [-0.7533,  0.4308],
           [ 0.7653, -1.0404]],

          [[-0.4184, -0.3899],
           [-0.5725, -1.2871],
           [-0.7463,  0.0388],
           [-0.4549,  0.0852]]]]], grad_fn=<StackBackward0>)
torch.Size([1, 2, 4, 4, 2])

```
