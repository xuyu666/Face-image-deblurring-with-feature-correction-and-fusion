import os
import sys
import math
from turtle import forward
import fire
import json

from functools import partial

import timm
from timm.models.layers import DropPath, trunc_normal_

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
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler as lr_scheduler

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

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import aim
from tensorboardX import SummaryWriter
import time

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'


import help_func
import help_class
from model_vgg import FeatureExtraction
from help_class import *
from help_func import *


# constants
NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png', 'JPEG']
img_files = []
table = True
global_a = True
rm_writer_file = True
num_test = 0

lr_reals = []

# stylegan2 classes

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False, device=0):
        super().__init__()
        self.rank = device
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            help_class.Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        global num_test

        b, c, h, w = x.shape
        # print(istyle.type())
        style = self.to_style(istyle)

        x = self.conv(x, style)

        if help_func.exists(prev_rgb):
            x = x + prev_rgb

        if help_func.exists(self.upsample):
            x = self.upsample(x)

        return x

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

''' 附加模块
class SenetBlock(nn.Module):
    def __init__(self, filters):
        super(SenetBlock, self).__init__()

        self.noise_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(filters,filters//16,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(filters//16,filters,kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.noise_se(input)

        return output

class NoiseFTBlock(nn.Module):
    def __init__(self, filters, act=True):
        super(NoiseFTBlock, self).__init__()
        self.act = act
        self.FT_se = SenetBlock(filters)
        self.FT_convMod = Conv2DMod(filters, filters, 3)
        
        if self.act:
            self.FT_act = nn.ReLU()     
    
    def forward(self, x):
        
        x_se = self.FT_se(x).squeeze()
        x = self.FT_convMod(x, x_se)
        if self.act:
            x = self.FT_act(x)

        return x

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        
class FT_Net(nn.Module):
    def __init__(self, filters, blocklayers=3, act = True):
        super(FT_Net, self).__init__()

        Ft_res = []
        for i in range(blocklayers):
            if i == blocklayers-1:
                act = False
            Ft_res.append(NoiseFTBlock(filters, act))

        self.FT_res = nn.Sequential(*Ft_res)

    def forward(self, input):
        output = self.FT_res(input)

        return output

'''

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Attn + sigmoid
class AttentionModule(nn.Module):
    def __init__(self, dim, act=True):
        super().__init__()
        self.act_choise = act
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        if self.act_choise:
            self.act = nn.Sigmoid()

    def forward(self, x):      
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        if self.act_choise:
            attn = self.act(attn) 

        return attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit_1 = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)


    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit_1(x)

        x = self.proj_2(x)
        x = x + shorcut
        return x


# VAN_完整Block
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=2., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        # self.norm = nn.GroupNorm(1, dim)

        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        
        # B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        # x = self.norm(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class Conv2d_Block(nn.Module):

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(Conv2d_Block, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class SenetBlock(nn.Module):
    def __init__(self, filters):
        super(SenetBlock, self).__init__()

        self.noise_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(filters,filters//16,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(filters//16,filters,kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.noise_se(input)

        return output


# Res_Block
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, strides=1, is_se=False, act=True):
        super(BasicBlock, self).__init__()
        self.is_se = is_se
        self.is_act = act
        self.conv1 = Conv2d_Block(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
        self.conv2 = Conv2d_Block(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
        if self.is_se:
            self.se = SenetBlock(out_channels)

        if self.is_act:
            self.act = nn.ReLU()


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + x

        if self.is_act:
            out = self.act(out)
        
        return out


# VAN_demo add in Res
class VAN_res_Block(nn.Module):

    def __init__(self, in_channels, out_channels, strides=1, is_se=False, act=True):
        super(VAN_res_Block, self).__init__()
        self.is_se = is_se
        self.is_act = act
        self.VAN_attn_1 = SpatialAttention(in_channels)
        self.conv1 = Conv2d_Block(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
        self.VAN_attn_2 = SpatialAttention(in_channels)
        self.conv2 = Conv2d_Block(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
        if self.is_se:
            self.se = SenetBlock(out_channels)

        if self.is_act:
            self.act = nn.ReLU()


    def forward(self, x):
        out = self.VAN_attn_1(x)
        out = self.conv1(out)
        out = self.VAN_attn_2(out)
        out = self.conv2(out)

        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
            
        out = out + x

        if self.is_act:
            out = self.act(out)
        
        return out


class Res_FT(nn.Module):
    def __init__(self, filters, FT_layers=2, is_se=True, act=True):
        super(Res_FT, self).__init__()

        FT_net = []
        for i in range(FT_layers):
            if i == FT_layers-1:
                act = False
            FT_net.append(BasicBlock(filters, filters, is_se=is_se, act=act))

        self.FT_net = nn.Sequential(*FT_net)

    def forward(self, input):
        output = self.FT_net(input)

        return output


# VAN 完整Block + Res_block_2
class VAN_Res(nn.Module):
    '''
    VAN 完整Block + Res_block_2 
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.VAN_attn = Block(in_channels)
        self.res = BasicBlock(in_channels, out_channels, act=False)

    def forward(self, x):
        x = self.VAN_attn(x)
        x = self.res(x)

        return x

# -------------------------------
class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, sft_half=True, device=0, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        self.input_channels = input_channels
        self.filters = filters
        self.rank = device

        # 双线性插值
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        # self.to_style1 = nn.Linear(latent_dim, input_channels)
        # self.to_noise1 = nn.Linear(1, filters)
        # self.style1_ca = help_class.eca_layer(input_channels)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        # self.to_style2 = nn.Linear(latent_dim, filters)
        # self.to_noise2 = nn.Linear(1, filters)
        # self.style2_ca = help_class.eca_layer(filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        # from Naf.Naf_add import NAFNet
        # self.conv3 = NAFNet(width=filters, enc_blk_nums=[18])
        # self.conv4 = NAFNet(width=filters, enc_blk_nums=[18])

        self.conv3 = help_class.attn_and_ff(filters)
        self.conv4 = help_class.attn_and_ff(filters)

        '''
        ===== for SFT modulations (scale and shift) =====      
        out_channels = filters
        if sft_half:
            sft_out_channels = out_channels
        else:
            sft_out_channels = out_channels * 2
        self.conv3_condition_scale = nn.Sequential(
                                nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                                nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1))
            
        self.conv3_condition_shift = nn.Sequential(
                                nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                                nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1))

        self.conv4_condition_scale = nn.Sequential(
                                nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                                nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1))
            
        self.conv4_condition_shift = nn.Sequential(
                                nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                                nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1))
        ===== for SFT modulations (scale and shift) ===== 

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(filters, filters, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(filters, filters, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(filters, filters, 3, padding=1)
        # )

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(filters, filters, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(filters, filters, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(filters, filters, 3, padding=1)
        # )

        # self.conv3 = help_class.attn_and_ff(filters)
        # self.conv4 = help_class.attn_and_ff(filters)

        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1)
        )

        # self.conv3 = VAN_res_Block(filters, filters, act=False)
        # self.conv4 = VAN_res_Block(filters, filters, act=False)

        # self.conv3 = Conv2DMod(filters, filters, 3)
        # self.conv4 = Conv2DMod(filters, filters, 3)

        # noise_conv
        # self.conv3 = nn.Conv2d(filters, filters, 3, padding=1)
        # self.conv4 = nn.Conv2d(filters, filters, 3, padding=1)

        # noise_SEnet
        # self.noise_se1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1,1)),
        #     nn.Conv2d(filters,filters//16,kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(filters//16,filters,kernel_size=1),
        #     nn.Sigmoid()
        # )

        # self.noise_se2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1,1)),
        #     nn.Conv2d(filters,filters//16,kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(filters//16,filters,kernel_size=1),
        #     nn.Sigmoid()
        # )

        # self.noise1_cbam = CBAM(channel=filters)
        # self.noise2_cbam = CBAM(channel=filters)

        self.noise1_FT = Res_FT(filters)
        self.noise2_FT = Res_FT(filters)
        '''
        
        self.activation = help_func.leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba, device=device)

    def forward(self, x, prev_rgb, istyle, inoise, noise_origin=False):
        batch, channels = istyle.shape
        if help_func.exists(self.upsample):
            x = self.upsample(x)

        b, c, x_chan, y_chan = x.shape
        if noise_origin == True:
            inoise = torch.zeros(b, self.filters, x_chan, y_chan).cuda(self.rank)
            noise1 = inoise
            noise2 = inoise

        else:
            noise1 = inoise[0]
            noise2 = inoise[1]

            # noise1 = self.noise_conv1(noise1)
            # noise2 = self.noise_conv2(noise2)
            noise1 = self.conv3(noise1)
            noise2 = self.conv4(noise2)


            '''
            # add SFT
            # noise1_scale = self.conv3_condition_scale(noise1)
            # noise1_shift = self.conv3_condition_shift(noise1)

            # noise2_scale = self.conv4_condition_scale(noise2)
            # noise2_shift = self.conv4_condition_shift(noise2)

            # noise1 = self.noise1_FT(inoise[0])
            # noise2 = self.noise2_FT(inoise[1])

            # noise1_attn = self.noise_se1(noise1)
            # noise2_attn = self.noise_se2(noise2)

            # noise1 = noise1 * noise1_attn
            # noise2 = noise2 * noise2_attn

            # noise1 = self.noise1_cbam(noise1)
            # noise2 = self.noise2_cbam(noise2)
            '''

        # style1 = self.to_style1(istyle)
        style1 = torch.ones(batch, self.input_channels).cuda(self.rank)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)
        # x = self.activation(x * noise1_scale + noise1_shift)

        # style2 = self.to_style2(istyle)
        style2 = torch.ones(batch, self.filters).cuda(self.rank)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)
        # x = self.activation(x * noise2_scale + noise2_shift)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        
        return x, rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            help_func.leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            help_func.leaky_relu()
        )

        self.downsample = nn.Sequential(
            help_class.Blur(),
            nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        # add D_layers
        # _, _, x_chan, y_chan = x.shape
        x = self.net(x)

        if help_func.exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, device=0, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)
        self.rank = device
        
        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        # map <-------> zip
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = list(zip(filters[:-1], filters[1:]))
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        # self.style_init = nn.Parameter(torch.randn((self.num_layers, self.latent_dim)))
        self.noise_init = nn.Parameter(torch.FloatTensor(image_size, image_size, 1).uniform_(0., 1.))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = help_class.attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent,
                device = device
            )
            
            self.blocks.append(block)

    def forward(self, vgg_img, vgg_feature, device=0):
        global num_test
        batch_size = vgg_img.shape[0]
        image_size = self.image_size

        styles = torch.ones(batch_size, self.num_layers, self.latent_dim).cuda(self.rank)
        # input_noise = help_func.image_noise(vgg_img, device=self.rank)
        input_noise = vgg_feature

        noise_ori = self.noise_init.expand(batch_size, -1, -1, -1).cuda(self.rank)
        # noise_ori = torch.ones_like(vgg_img)

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        kk = 0
        noise_layer = 0
        for style, block, attn in zip(styles, self.blocks, self.attns):
            count = kk*2
            kk+=1
            if help_func.exists(attn):
                x = attn(x)

            # x, rgb = block(x, rgb, style, input_noise[count:count+2], noise_origin=True)
            # noise_layer += 2

            if noise_layer < 8:
                x, rgb = block(x, rgb, style, input_noise[count:count+2])
                noise_layer += 2
            else:
                x, rgb = block(x, rgb, style, input_noise[count:count+2], noise_origin=True)
                noise_layer += 2

        return rgb

class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, fq_layers = [], fq_dict_size = 256, attn_layers = [], transparent = False, fmap_max = 512):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        # filters = [num_init_filters] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]
        filters = [num_init_filters] + [(network_capacity * 2) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        # writeToFile('D_filters', filters)
        # writeToFile('D_channels', chan_in_out)

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = help_class.attn_and_ff(out_chan) if num_layer in attn_layers else None
            attn_blocks.append(attn_fn)

            quantize_fn = help_class.PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if str(num_layer) in str(fq_layers) else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = help_class.Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if help_func.exists(attn_block):
                x = attn_block(x)

            if help_func.exists(q_block):
                x, _, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss

class StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim = 512, fmap_max = 512, style_depth = 8, network_capacity = 16, transparent = False, fp16 = False, cl_reg = False, steps = 1, lr = 1e-4, ttur_mult = 2, fq_layers = [], fq_dict_size = 256, attn_layers = [], no_const = False, lr_mlp = 0.1, rank = 0):
        super().__init__()
        global table
        self.lr = lr
        self.steps = steps
        # self.ema_updater = EMA(0.995)
        self.rank = rank

        # self.S = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, device=rank, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers = fq_layers, fq_dict_size = fq_dict_size, attn_layers = attn_layers, transparent = transparent, fmap_max = fmap_max)

        # self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        # self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const)

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = help_class.AugWrapper(self.D, image_size)

        if table:
            help_func.writeToFile(clear=True)
            help_func.writeToFile('G_Layers', self.G)
            help_func.writeToFile('D_Layers', self.D)
            table = False

        # turn off grad for exponential moving averages
        # set_requires_grad(self.SE, False)
        # set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters())
        self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))
        # optimizer学习率初值为2e-4
        # self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_opt, T_max=10, eta_min=2e-5)
        # self.D_scheduler = lr_scheduler.CosineAnnealingLR(self.D_opt, T_max=10, eta_min=3e-5)

        self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_opt, T_max=15, eta_min=2e-5)
        self.D_scheduler = lr_scheduler.CosineAnnealingLR(self.D_opt, T_max=15, eta_min=2e-5 * ttur_mult)

        # self.G_scheduler = lr_scheduler.ExponentialLR(self.G_opt, gamma=0.8)
        # self.D_scheduler = lr_scheduler.ExponentialLR(self.D_opt, gamma=0.8)


        # init weights
        self._init_weights()
        # self.reset_parameter_averaging()

        self.cuda(rank)

        # startup apex mixed precision
        self.fp16 = fp16
        if fp16:
            # (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)
            (self.G, self.D), (self.G_opt, self.D_opt) = amp.initialize([self.G, self.D], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        # for block in self.G.blocks:
        #     nn.init.zeros_(block.to_noise1.weight)
        #     nn.init.zeros_(block.to_noise2.weight)
        #     nn.init.zeros_(block.to_noise1.bias)
        #     nn.init.zeros_(block.to_noise2.bias)

    def forward(self, x):
        return x
