import os
from pyexpat import model
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
from torch.optim import Adam
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
from datetime import datetime
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import aim
from tensorboardX import SummaryWriter
import time
import scipy.io as scio
import random as random_new

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

from model_vgg import FeatureExtraction
import model
from model import *
import help_func
from help_func import *

# constants
NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png', 'JPEG', 'mat']
img_files = []
table = True
global_a = True
rm_writer_file = True

lr_reals = []

def recursive_listdir(path, new=False):

    global img_files

    if new:
        img_files = []
        new = False
    files = os.listdir(path)
    file_path = []
    img_file = []
    for file in files:
        file_path = os.path.join(path, file)

        if os.path.isfile(file_path):
            img_file.append(file_path)

        elif os.path.isdir(file_path):
            recursive_listdir(file_path)

    img_files.extend(img_file)
    return img_files


# helper classes
# Adam_init
class Adam_init(Optimizer):
    #region
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    #endregion

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_init, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            # F.adam(params_with_grad,
            lr_params = []
            lr_params = adam(params_with_grad,
                                grads,
                                exp_avgs,
                                exp_avg_sqs,
                                max_exp_avg_sqs,
                                state_steps,
                                group['amsgrad'],
                                beta1,
                                beta2,
                                group['lr'],
                                group['weight_decay'],
                                group['eps'])
        return lr_params, loss

class PerceptualLoss():
	
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def __init__(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

class Residual(nn.Module):
    def __init__(self, fn, act=False):
        super().__init__()
        self.is_act = act
        self.fn = fn
        if self.is_act:
            self.act = help_func.leaky_relu()
        
    def forward(self, x):
        x = self.fn(x) + x
        if self.is_act:
            x = self.act(x)

        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, bn=False):
        super().__init__()
        self.fn = fn
        if bn:
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        # std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        # return (x - mean) / (std + self.eps) * self.g + self.b
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not help_func.exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

# dataset
class G_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, path1, kernel_path='/data'):
        # 获取图片路径
        global img_files
        img_files = []
        fname_paths_1 = []
        fname1 = []

        if fname1 in sorted(os.listdir(path1)):
            for fname1 in sorted(os.listdir(path1)):  
                init_path = os.path.join(path1, fname1)
                fname_paths_1 = recursive_listdir(init_path)
        else:
            fname_paths_1 = recursive_listdir(path1)

        self.sharp_path = fname_paths_1
        self.kernel_paths = [p for p in Path(f'{kernel_path}').glob(f'**/*.mat')]

    def __getitem__(self, index):
        
        sharp_image = Image.open(self.sharp_path[index])
        sharp_image = help_func.transform(sharp_image)
        
        sharp_image_cv = cv2.imread(self.sharp_path[index])
        kernel_tensor = self.random_load_kernel()

        blur_image = self.blur(sharp_image_cv, kernel_tensor)
        blur_image = help_func.transform(blur_image)

        return blur_image, sharp_image

    def blur(self, input_cpu_cv, kernel_tensor):
        input_cpu_cv = input_cpu_cv/255.0
        input_cpu_cv = cv2.filter2D(input_cpu_cv, -1, kernel_tensor, borderType=cv2.BORDER_REPLICATE)

        input_cpu = input_cpu_cv + 0.01 * np.float32(np.random.randn(*(input_cpu_cv.shape)))
        # clip范围
        input_cpu = np.clip(input_cpu, 0, 1)  
        input_cpu = np.uint8(input_cpu*255)

        input_Image = self.cv_to_Img(input_cpu)
        return input_Image

    def cv_to_Img(self, input_cv):
        input_Image = Image.fromarray(cv2.cvtColor(input_cv,cv2.COLOR_BGR2RGB))  
        return input_Image

    def random_load_kernel(self):
        kernel_index = random_new.randint(0, len(self.kernel_paths)-1)
        kernel_path = self.kernel_paths[kernel_index]
        kernel_middle = scio.loadmat(kernel_path)
        kernel_tensor = kernel_middle['kernel']

        return kernel_tensor

    def tensor2numpy(self, input_tensor: torch.Tensor):
        input_tensor=input_tensor.numpy()
        in_arr=np.transpose(input_tensor,(1,2,0))#将(c,w,h)转换为(w,h,c)。但此时如果是全精度模型，转化出来的dtype=float64 范围是[0,1]。后续要转换为cv2对象，需要乘以255
        cv_img=cv2.cvtColor(np.uint8(in_arr*255), cv2.COLOR_RGB2BGR)

        return cv_img

    def __len__(self):
        return len(self.sharp_path)

class G_Dataset_val(torch.utils.data.Dataset):
    
    def __init__(self, path1):
        # 获取图片路径
        global img_files
        img_files = []
        fname_paths_1 = []
        fname1 = []

        if fname1 in sorted(os.listdir(path1)):
            for fname1 in sorted(os.listdir(path1)):  
                init_path = os.path.join(path1, fname1)
                fname_paths_1 = recursive_listdir(init_path)
        else:
            fname_paths_1 = recursive_listdir(path1)

        self.sharp_path = fname_paths_1

    def __getitem__(self, index):
        sharp_image = Image.open(self.sharp_path[index])
        sharp_image = help_func.transform(sharp_image)

        sharp_path_output = self.sharp_path[index]
        return sharp_image, sharp_path_output

    def __len__(self):
        return len(self.sharp_path)

class MyDataset_kernel(torch.utils.data.Dataset):
    
    def __init__(self, path1):
        # 获取图片路径
        global img_files
        img_files = []
        fname_paths_1 = []
        fname1 = []

        if fname1 in sorted(os.listdir(path1)):
            first = True
            for fname1 in sorted(os.listdir(path1)):  
                if first:
                    init_path = os.path.join(path1, fname1)
                    fname_paths_1 = recursive_listdir(init_path, new=True)
                else:
                    init_path = os.path.join(path1, fname1)
                    fname_paths_1 = recursive_listdir(init_path)
        else:
            fname_paths_1 = recursive_listdir(path1, new=True)

        self.kernel_path = fname_paths_1

    def __getitem__(self, index):
        kernel_path = self.kernel_path[index]

        # kernel_middle = scio.loadmat(kernel_path)
        # kernel_tensor = kernel_middle['kernel']

        # padding = np.int((31 - kernel_tensor.shape[0])/2)
        # kernel_tensor = np.lib.pad(kernel_tensor, padding, mode='constant', constant_values = 0)

        return kernel_path

    def __len__(self):
        return len(self.kernel_path)

class D_Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent = False, aug_prob = 0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = help_func.transform_d(img)
        # img = help_func.transform_1(img)

        # path = self.paths[index]
        # img = cv2.imread(str(path))
        # img = test_4.ImageRotate_d(img)

        # image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        # img = help_func.transform_d(image)

        return img

class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, path1, path2):
        # 获取图片路径
        global img_files
        count = 0
        kk =0 
        img_files = []
        fname_paths_1 = []
        fname_paths_2 = []
        fname1 = []
        fname2=[]

        if fname1 in sorted(os.listdir(path1)):
            for fname1 in sorted(os.listdir(path1)):  
                init_path = os.path.join(path1, fname1)
                fname_path_1 = recursive_listdir(init_path)
        else:
            fname_path_1 = recursive_listdir(path1)

        img_files = []

        for kk in range(len(fname_path_1)):
            fname_paths_1.append(fname_path_1[kk])

            img_num1 = fname_path_1[kk].split('/')[-2]
            img_num2 = fname_path_1[kk].split('/')[-1]

            # img = os.path.join(path2, img_num1, img_num2)
            img = os.path.join(path2, img_num2)
            fname_paths_2.append(img)
            kk += 1

        self.blur_path = fname_paths_1
        self.sharp_path = fname_paths_2

    def __getitem__(self, index):
        blur_image = Image.open(self.blur_path[index])
        sharp_image = Image.open(self.sharp_path[index])
        blur_image = help_func.transform(blur_image)
        sharp_image = help_func.transform(sharp_image)

        return blur_image, sharp_image

    def __len__(self):
        return len(self.blur_path)

# attention

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):

        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        # k转置 * v
        context = einsum('b n d, b n e -> b d e', k, v)
        # q * (k转置 * v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

# noise_add
class SFTLayer(nn.Module):
    def __init__(self, channels):
        super(SFTLayer, self).__init__()
        self.channels = channels
        self.SFT_scale_conv0 = nn.Conv2d(self.channels, self.channels*2, 1)
        self.SFT_scale_conv1 = nn.Conv2d(self.channels*2, self.channels, 1)
        self.SFT_shift_conv0 = nn.Conv2d(self.channels, self.channels*2, 1)
        self.SFT_shift_conv1 = nn.Conv2d(self.channels*2, self.channels, 1)

    def forward(self, x):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x), 0.1, inplace=True))
        return x * (scale + 1) + shift


class Block_SFT(nn.Module):
    def __init__(self, channels):
        super(Block_SFT, self).__init__()
        self.channels = channels
        self.sft0 = SFTLayer(self.channels)
        self.conv0 = nn.Conv2d(self.channels, self.channels, 3, 1, 1)
        self.sft1 = SFTLayer(self.channels)
        self.conv1 = nn.Conv2d(self.channels, self.channels, 3, 1, 1)

    def forward(self, x):
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1(fea)
        # fea = self.conv1(fea)
        fea = F.relu(self.conv1(fea), inplace=True)
        return fea 


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

        # self.conv_mod = model.Conv2DMod(channel, channel, 3)

    def forward(self, x):
        input = x.clone()
        # feature descriptor on the global spatial information
        y = self.avg_pool(input)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        # Ori
        # return x * y.expand_as(x)

        # 利用通道注意力调制权重
        output = y.squeeze()
        # output = self.conv_mod(x, y)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# -------------------------------
def _make_divisible(v, divisor, min_value=None):
    """
    这个函数的目的是确保Channel能被8整除。
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
 

class MBConv(nn.Module):
    """
     定义MBConv模块和Fused-MBConv模块，将fused设置为1或True是Fused-MBConv，否则是MBConv
    :param inp:输入的channel
    :param oup:输出的channel
    :param stride:步长，设置为1时图片的大小不变，设置为2时，图片的面积变为原来的四分之一
    :param expand_ratio:放大的倍率
    :return:
    """
    def __init__(self, inp, oup, stride, expand_ratio=4, fused=False):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if fused:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
 
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
 
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

# one layer of self-attention and feedforward, for images

# ===== Double =====
attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(PreNorm(chan, LinearAttention(chan))),
    Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1), help_func.leaky_relu(), nn.Conv2d(chan * 2, chan, 1)))),
    Residual(PreNorm(chan, LinearAttention(chan))),
    Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1), help_func.leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])

# ===== Single =====
# attn_and_ff = lambda chan: nn.Sequential(*[
#     Residual(PreNorm(chan, LinearAttention(chan))),
#     Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1), help_func.leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
# ])

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False):
        # if random() < prob:
        #     images = random_hflip(images, prob=0.5)
        #     images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)

if __name__ == '__main__':

    a = ResBlock_SFT()

    tensor_test = torch.randn(1, 64, 4, 4)
    tensor_test_1 = torch.randn(1, 32, 4, 4)

    demo_1 = (tensor_test, tensor_test_1)

    result = a(demo_1)

    print('-------------')