import numpy as np
from torchvision import transforms
from PIL import Image
from torch import nn
import torch
import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):

        hr_height, hr_width = hr_shape
        
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        #print('root: ', root)
        self.files = sorted(glob.glob(root + "/*.*"))
        #print('self.files: ', self.files)
        self.files = self.files[0:opt.nrOfImages]

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)