from turtle import forward
from sqlalchemy import false, true
from torch import nn
import torch
import json
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.models import vgg19

from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from .utils import extract_archive, verify_str_arg, iterable_to_str
from .vision import VisionDataset
from PIL import Image



class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()

        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
            return self.feature_extractor(img)   

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_features, 0.8)
        self.prelu = nn.PReLU()

    def forward(self, x):
        xin = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        return xin + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()

        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        
        self.res_block = nn.Sequential(*res_blocks)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, 0.8)

        upsampling = []

        for out_features in range(2):
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]

        self.upsampling = nn.Sequential(*upsampling)
        
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()


    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.res_block(out1)

        out2 = self.bn2(self.conv2(out))

        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.tanh(self.conv3(out))

        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape

        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            return layers     

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class opt:
    epoch = 0
    n_epochs = 5
    dataset_name = "img_align_celeba" #https://www.kaggle.com/jessicali9530/celeba-dataset/activity
    batch_size = 4

    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    n_cpu = 8
    
    hr_height = 256
    hr_width = 256
    channels = 3
    
    sample_interval = 100
    checkpoint_interval = 2
    nrOfImages = 10000