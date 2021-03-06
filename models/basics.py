import torch
import numpy
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, PReLU
from torch.nn.modules.batchnorm import BatchNorm2d

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_channel, kernel, stride, pd):
        super().__init__()        
        self.conv1 = nn.Conv2d(in_ch, out_channel, kernel, stride, pd)
        self.bn1 = BatchNorm2d(out_channel)
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv2d(in_ch, out_channel, kernel, stride, pd)
        self.bn2 = BatchNorm2d(out_channel)

        list = [self.conv1, self.bn1, self.prelu, self.conv2, self.bn2]
        self.layers = nn.ModuleList(list)

    def forward(self, x):
        temp = x
        for layer in self.layers:
            x = layer(x)
        x += temp
        return x


class postBlock(nn.Module):
    def __init__(self, in_ch, out, kernel, stride,pd):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out, kernel, stride,pd)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

        list = [self.conv, self.pixelshuffle, self.prelu]

        self.layers = nn.ModuleList(list)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out, kernel, stride, pd, slope) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out, kernel, stride, pd)
        self.bn = BatchNorm2d(out)
        self.leaky = nn.LeakyReLU(slope)

        list = [self.conv, self.bn, self.leaky]

        self.layers = nn.ModuleList(list)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, 1, 4)
        self.prelu1 = nn.PReLU()
        self.pre_conv = nn.Sequential(self.conv1, self.prelu1)

        resi_layers = [ResidualBlock(64,64,3,1,1) for _ in range(16)]
        self.resi_blocks = nn.Sequential(*resi_layers)


        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.bn = nn.BatchNorm2d(64)
        self.post_conv = nn.Sequential(self.conv2, self.bn)
        
        post_layers = [postBlock(64,256,3,1,1) for _ in range(2)]
        self.postblocks = nn.Sequential(*post_layers)

        self.conv3 = nn.Conv2d(64,3,9,1,4)

    def forward(self, x):

        x = self.pre_conv(x)
        temp = x
        x = self.resi_blocks(x)
        x = self.post_conv(x)
        x += temp
        x = self.postblocks(x)
        x = self.conv3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.leaky = nn.LeakyReLU(0.2)

        self.pre_conv = nn.Sequential(self.conv1, self.leaky)

        self.convB1 = ConvBlock(64,64,3,2,1,0.2)
        self.convB2 = ConvBlock(64,128,3,1,1,0.2)
        self.convB3 = ConvBlock(128,128,3,2,1,0.2)
        self.convB4 = ConvBlock(128,256,3,1,1,0.2)
        self.convB5 = ConvBlock(256,256,3,2,1,0.2)
        self.convB6 = ConvBlock(256,512,3,1,1,0.2)
        self.convB7 = ConvBlock(512,512,3,2,1,0.2)

        list = [self.convB1,self.convB2,self.convB3,self.convB4,self.convB5,self.convB6,self.convB7]

        self.conv_block = nn.Sequential(*list)
        self.ln1 = nn.Linear(512*6*6,1024)
        self.ln2 = nn.Linear(1024,1)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.conv_block(x)
        x = self.ln1(x.view(x.size(0), -1))
        x = self.leaky(x)
        x = self.ln2(x)
        return torch.sigmoid(x)




