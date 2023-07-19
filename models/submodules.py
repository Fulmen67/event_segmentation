"""
Adapted from TUDelft-MAVLab https://github.com/tudelft/event_flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

from utils.LeakyDoReLU import LeakyDoReLU

class Linear(nn.Module):
    """
    Linear layer.
    Default: bias, leaky relu, no downsampling, no batch norm.
    
    """
    def __init__(self, in_features, out_features, activation="LeakyReLU", norm=None, BN_momentum=0.1, w_scale=None):
        super(Linear, self).__init__()
        bias = False if norm == "BN" else True
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if w_scale is not None:
            nn.init.uniform_(self.linear.weight, -w_scale, w_scale)
            nn.init.zeros_(self.linear.bias)

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
            elif hasattr(nn, activation):
                self.activation = getattr(nn, activation)()
        else:
            self.activation = None

        if norm == "BN":
            self.norm = nn.BatchNorm1d(out_features, momentum=BN_momentum)
        else:
            self.norm = None
    
    def forward(self, x): 
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)        
        return x
    
     
class ConvLayer(nn.Module):
    """
    Convolutional layer.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
        BN_momentum=0.1,
        w_scale=None,
    ):
        super(ConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if w_scale is not None:
            nn.init.uniform_(self.conv2d.weight, -w_scale, w_scale)
            nn.init.zeros_(self.conv2d.bias)

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
            elif hasattr(nn, activation):
                self.activation = getattr(nn, activation)()
            
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class ConvLayer_(ConvLayer):
    """
    Clone of ConvLayer that acts like it has state, and allows residual.
    """

    def forward(self, x, prev_state, residual=0):
        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.tensor(0)  # not used

        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        out += residual
        if self.activation is not None:
            out = self.activation(out)

        return out, prev_state


class TransposedConvLayer(nn.Module):
    """
    Transposed convolutional layer to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation="relu",
        norm=None,
    ):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=bias,
        )

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    """
    Upsampling layer (bilinear interpolation + Conv2d) to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
    ):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
            elif activation == "leaky_dorelu":
                self.activation = LeakyDoReLU(gamma = 100)
            elif hasattr(nn, activation):
                self.activation = getattr(nn, activation)()  
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out
    
class ResidualBlock(nn.Module):
    """
    Residual block as in "Deep residual learning for image recognition", He et al. 2016.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation="relu",
        downsample=None,
        norm=None,
        BN_momentum=0.1,
    ):
        super(ResidualBlock, self).__init__()
        bias = False if norm == "BN" else True
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.bn1 = nn.InstanceNorm2d(out_channels, track_running_stats=True)
            self.bn2 = nn.InstanceNorm2d(out_channels, track_running_stats=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        if self.norm in ["BN", "IN"]:
            out1 = self.bn1(out1)

        if self.activation is not None:
            out1 = self.activation(out1)

        out2 = self.conv2(out1)
        if self.norm in ["BN", "IN"]:
            out2 = self.bn2(out2)

        if self.downsample:
            residual = self.downsample(x)

        out2 += residual
        if self.activation is not None:
            out2 = self.activation(out2)

        return out2, out1