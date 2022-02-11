from torch import nn
import torch

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
