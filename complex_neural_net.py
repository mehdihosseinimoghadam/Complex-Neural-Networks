"""
Complex Valued Neural Layers From Scratch
Programmed by Mehdi Hosseini Moghadam
*    MIT Licence
*    2022-02-15 Last Update
"""






from torch import nn
import torch


##__________________________________Complex Linear Layer __________________________________________


class CLinear(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(CLinear, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.re_linear = nn.Linear(self.in_channels, self.out_channels, **kwargs)
    self.im_linear = nn.Linear(self.in_channels, self.out_channels, **kwargs)

    nn.init.xavier_uniform_(self.re_linear.weight)
    nn.init.xavier_uniform_(self.im_linear.weight)



  def forward(self, x):  
    x_re = x[..., 0]
    x_im = x[..., 1]

    out_re = self.re_linear(x_re) - self.im_linear(x_im)
    out_im = self.re_linear(x_im) + self.im_linear(x_re)

    out = torch.stack([out_re, out_im], -1) 

    return out
  
  

##______________________________________Complex Convolution 2d_____________________________________________

class CConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(CConv2d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels


    self.re_conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)
    self.im_conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)

    nn.init.xavier_uniform_(self.re_conv.weight)
    nn.init.xavier_uniform_(self.im_conv.weight)

  def forward(self, x):  
    x_re = x[..., 0]
    x_im = x[..., 1]

    out_re = self.re_conv(x_re) - self.im_conv(x_im)
    out_im = self.re_conv(x_im) + self.im_conv(x_re)

    out = torch.stack([out_re, out_im], -1) 

    return out

  
  ##___________________________________Complex Convolution Transpose 2d_______________________________________________
  
  
class CConvTrans2d(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(CConvTrans2d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels


    self.re_Tconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, **kwargs)
    self.im_Tconv = nn.ConvTranspose2d(self.in_channels, self.out_channels, **kwargs)

    nn.init.xavier_uniform_(self.re_Tconv.weight)
    nn.init.xavier_uniform_(self.im_Tconv.weight)


  def forward(self, x):  
    x_re = x[..., 0]
    x_im = x[..., 1]

    out_re = self.re_Tconv(x_re) - self.im_Tconv(x_im)
    out_im = self.re_Tconv(x_im) + self.im_Tconv(x_re)

    out = torch.stack([out_re, out_im], -1) 

    return out
      
  
  ##___________________________Complex BatchNorm Layer____________________________________
  
  
class CBatchnorm(nn.Module):
  def __init__(self, in_channels):
      super(CBatchnorm, self).__init__()
      self.in_channels = in_channels

      self.re_batch = nn.BatchNorm2d(in_channels)
      self.im_batch = nn.BatchNorm2d(in_channels)


  def forward(self, x):
      x_re = x[..., 0]
      x_im = x[..., 1]

      out_re =  self.re_batch(x_re)
      out_im =  self.re_batch(x_im)


      out = torch.stack([out_re, out_im], -1) 

      return out

##_______________________Complex Convolutional Block_______________________________________


class CconvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(CconvBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.CConv2d = CConv2d(self.in_channels, self.out_channels, **kwargs)
    self.CBatchnorm = CBatchnorm(self.out_channels)
    self.leaky_relu = nn.LeakyReLU()


  def forward(self, x):
    conved = self.CConv2d(x)
    normed = self.CBatchnorm(conved)
    activated =  self.leaky_relu(normed)

    return activated
  
  
  ##__________________________________Complex Convolutional Transpose Block________________________________________
  
  
class CConvTransBlock(nn.Module):
  def __init__(self, in_channels, out_channels, last_layer=False, **kwargs):
    super(CConvTransBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.last_layer = last_layer

    self.CConvTrans2d = CConvTrans2d(self.in_channels, self.out_channels, **kwargs)
    self.CBatchnorm = CBatchnorm(self.out_channels)
    self.leaky_relu = nn.LeakyReLU()


  def forward(self, x):
    conved =  self.CConvTrans2d(x)

    if not self.last_layer:
        normed = self.CBatchnorm(conved)
        activated =  self.leaky_relu(normed)
        return activated
    else:
        m_phase = conved/(torch.abs(conved)+1e-8)  
        m_mag = torch.tanh(torch.abs(conved))
        out = m_phase * m_mag
        return out  


##______________________Complex LSTM Layer_________________________________________________


class CLSTM(nn.Module):
  def __init__(self, in_channels, hidden_size, num_layers, **kwargs):
    super(CLSTM, self).__init__()
    self.in_channels = in_channels
    self.hidden_size = hidden_size
    self.num_layers = num_layers


    self.re_LSTM = nn.LSTM(self.in_channels, self.hidden_size, self.num_layers , **kwargs)
    self.im_LSTM = nn.LSTM(self.in_channels, self.hidden_size, self.num_layers, **kwargs)


  def forward(self, x, h0, c0):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re1, (hn_re1, cn_re1) =  self.re_LSTM(x_re, (h0[...,0], c0[...,0]))
        out_re2, (hn_re2, cn_re2) =  self.im_LSTM(x_im, (h0[...,1], c0[...,1]))
        out_re = out_re1 - out_re2
        hn_re  = hn_re1  - hn_re2
        cn_re  = cn_re1  - cn_re2

        out_im1, (hn_im1, cn_im1) =  self.re_LSTM(x_re, (h0[...,1], c0[...,1]))
        out_im2, (hn_im2, cn_im2) =  self.im_LSTM(x_im, (h0[...,0], c0[...,0]))
        out_im = out_im1 + out_im2
        hn_im  = hn_im1  + hn_im2
        cn_im  = cn_im1  + cn_im2

        out = torch.stack([out_re, out_im], -1) 
        hn = torch.stack([hn_re, hn_im], -1) 
        cn = torch.stack([cn_re, cn_im], -1) 

        return out, (hn, cn)

      
##_______________________________Complex MaxPooling 2d Layer___________________      


class CMaxPool2d(nn.Module):
  def __init__(self, kernel_size, **kwargs):
    super(CMaxPool2d, self).__init__()
    self.kernel_size = kernel_size


    self.CMax_re = nn.MaxPool2d(self.kernel_size, **kwargs)
    self.CMax_im = nn.MaxPool2d(self.kernel_size, **kwargs) 

  def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.CMax_re(x_re)
        out_im = self.CMax_im(x_im)


        out = torch.stack([out_re, out_im], -1) 

        return out

##________________________________Complex Average Pooling 2d Layer_____________________________ 

class CAvgPool2d(nn.Module):
  def __init__(self, kernel_size, **kwargs):
    super(CAvgPool2d, self).__init__()
    self.kernel_size = kernel_size


    self.CMax_re = nn.AvgPool2d(self.kernel_size, **kwargs)
    self.CMax_im = nn.AvgPool2d(self.kernel_size, **kwargs) 

  def forward(self, x):
        x_re = x[..., 0]
        x_im = x[..., 1]

        out_re = self.CMax_re(x_re)
        out_im = self.CMax_im(x_im)


        out = torch.stack([out_re, out_im], -1) 

        return out
