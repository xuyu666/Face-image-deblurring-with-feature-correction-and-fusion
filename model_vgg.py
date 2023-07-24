import os
import sys
import math
import fire
import json

# Adam优化器需要的包
from torch.optim.optimizer import Optimizer
# from torch.optim import _functional as F
from torch import Tensor
from typing import List, Optional
# Adam优化器需要的包
import cv2
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
# from skimage.measure import compare_ssim
from datetime import datetime

# from torchsummary import summary
import csv

from tqdm import tqdm
from math import floor, log2
from random import random
import shutil
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack


from sklearn import preprocessing 
import numpy as np

import torch
from torch import nn, einsum
from torch.utils import data
# from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange, repeat
from kornia.filters import filter2d

import torchvision
from torchvision import transforms, models
from torchvision import transforms, datasets, utils
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, Normalize, CenterCrop
from version import __version__
# from diff_augment import DiffAugment

from vector_quantize_pytorch import VectorQuantize

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

###VGG提取特征
# @torch.no_grad()
class VGG_custom(nn.Module):

    def __init__(self, features=None):
        super(VGG_custom, self).__init__()
        self.conv1 = features[0]   
        self.conv2 = features[1] 
        self.conv3 = features[2]
        self.conv4 = features[3]
        self.conv5 = features[4]
        self.conv6 = features[5]
        self.conv7 = features[6]
        self.conv8 = features[7]
        self.conv9 = features[8]
        self.conv10 = features[9]
        self.conv11 = features[10]

    def imshow(self, tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        unloader = transforms.ToPILImage()
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)

        noise = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x11]
        # noise = [x4, x5, x6, x7, x8, x9, x10, x11, x11]
           
        return noise

@torch.no_grad()
def FeatureExtraction():
    vgg13 = torchvision.models.vgg13()
    vgg13.load_state_dict(torch.load('models/vgg13.pth'))
    vgg13 = vgg13.features

    vgg_features = []
    layers = []
    kernel = [64, 64, 128, 128, 256, 256, 512, 512, 512, 512]
    feature_map = [128, 128, 64, 64, 32, 32, 16, 16, 8, 8]
    i = 2
    for name, layer in vgg13._modules.items():
        if isinstance(layer, nn.Conv2d):
            layers += [layer, nn.LayerNorm([kernel[i-2], feature_map[i-2],
                                            feature_map[i-2]]), nn.LeakyReLU(0.2, inplace=True)]
            vgg_features.append(nn.Sequential(*layers))
            layers = []
            i += 1
        elif isinstance(layer, nn.MaxPool2d):
            layers += [layer]
            if name == '24':
                vgg_features.append(nn.Sequential(layers[0]))
        else:
            continue
    
    vgg_custom = VGG_custom(vgg_features)
    
    return vgg_custom

import help_func
from help_func import *
import torch
import os

if __name__ == '__main__':
    img_path = '/workspace/xuyu/xuyu_train/xuyu_code/test/test_img/test_img_1/result/001815.png'
    result_path = '/workspace/xuyu/xuyu_train/xuyu_code/test/test_img/test_img_1/result/vgg/'
    os.makedirs(result_path, exist_ok=True)

    model_vgg = FeatureExtraction().cuda()

    img_tensor = Image.open(img_path)

    
    img_tensor = help_func.transform(img_tensor).cuda()
    img_tensor = torch.unsqueeze(img_tensor, 0)
    noise_vgg = model_vgg(img_tensor)

    for i in range (len(noise_vgg)):
        save_tensor_ori = noise_vgg[i]
        save_path_pre = os.path.join(result_path, str(i))
        os.makedirs(save_path_pre, exist_ok=True)

        for j in range(12):
        
            save_tensor_middle = save_tensor_ori[0, j, :, :]
            save_path = os.path.join(save_path_pre, str(j)+'.png')
            print(save_path)

            torchvision.utils.save_image(save_tensor_middle, save_path)

    print('------------------------------')

