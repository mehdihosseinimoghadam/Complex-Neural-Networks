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
