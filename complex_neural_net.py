from torch import nn
import torch



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

    out = torch.cat([out_re, out_im], -1) 

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

    out = torch.cat([out_re, out_im], -1) 

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


        out = torch.cat([out_re, out_im], -1) 

        return out

