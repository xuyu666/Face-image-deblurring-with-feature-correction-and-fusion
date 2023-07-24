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
import torch.nn.functional as f
from torch.fft import rfftn, irfftn

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

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import aim
from tensorboardX import SummaryWriter
import time

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

import model_vgg
from model_vgg import FeatureExtraction
from model import *
from help_class import *

import scipy.io as scio
import re
from scipy import signal
import random
import os

model_vgg = model_vgg.FeatureExtraction().cuda()

# constants
NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png', 'JPEG']
img_files = []
table = True
global_a = True
rm_writer_file = True

lr_reals = []

# helpers
# def gram_matrix(y):
#     """ Returns the gram matrix of y (used to compute style loss) """
#     (b, c, h, w) = y.size()
#     features = y.view(b, c, w * h)
#     features_t = features.transpose(1, 2)   #C和w*h转置
#     gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t相乘
#     return gram

# def gram_matrix(style_features, target_features):
#     _, channel, height, width = style_features.size()
#     size = height*width*channel

#     # 这里的reshape操作先将 特征图 变为两维， （height*width，channel），再计算 gram matrix
#     style_features = torch.reshape(style_features, (-1, channel))
#     style_gram = torch.matmul(style_features.transpose(1, 0), style_features)/size

#     target_features = torch.reshape(target_features, (-1, channel))
#     target_gram = torch.matmul(target_features.transpose(1, 0), target_features) / size

#     return nn.MSELoss((target_gram - style_gram))/size

def gram_matrix(x):
    shape_x = x.shape
    b = shape_x[0]
    c = shape_x[1]
    x = torch.reshape(x, [b, c, -1])
    # print(torch.matmul(x.transpose(2, 1), x))
    return torch.matmul(x, x.transpose(2, 1))

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())

    out = (im.astype('float') - min_val) / (max_val - min_val)

    return out

# a function to generate blurry images y: y = x * w + n 
def GenerateBlurry(x, w, noise_std):
    '''
    x = Image To Be BLURRED
    w = BLUR KERNEL
    '''
    channels = x.shape[-1]
    padding = np.int((x.shape[0] - w.shape[0])/2)
    w = np.lib.pad(w, padding, mode='constant', constant_values = 0)
    w = np.pad(w,((1,0),(1,0)),'constant',constant_values = (0,0))
    
    # applying blur kernel on each channel in freq. domain
    y_f = np.zeros(x.shape, dtype=complex)
    for i in range(channels):
        y_f[:,:,i] = np.fft.fft2(x[:,:,i]) * np.fft.fft2(w)
        
    # converting to spatial domain
    y = np.zeros(x.shape)
    for i in range(channels):
        y[:,:,i] = np.fft.fftshift( np.fft.ifft2(y_f[:,:,i]).real)
    
    # adding noise
    # noise = np.random.normal(0,noise_std, size = y.shape)
    # y = y + noise
    # y = np.clip(y,0,1)

    # y = np.uint8(y*255)
    # cv2.imwrite('./test_1.png', y)

    return y

##   u代表原矩阵，shiftnum1代表行，shiftnum2代表列。
def circshift(u,shiftnum1,shiftnum2):
    h,w = u.shape
    if shiftnum1 < 0:
        u = np.vstack((u[-shiftnum1:,:],u[:-shiftnum1,:]))
    else:
        u = np.vstack((u[(h-shiftnum1):,:],u[:(h-shiftnum1),:]))
    if shiftnum2 > 0:
        u = np.hstack((u[:, (w - shiftnum2):], u[:, :(w - shiftnum2)]))
    else:
        u = np.hstack((u[:,-shiftnum2:],u[:,:-shiftnum2]))
    return u

def GenerateBlurry_self_1(x, w, noise_std):
    '''
    x = Image To Be BLURRED
    w = BLUR KERNEL
    '''
    channels = x.shape[-1]
    big_v = np.zeros((128, 128))

    gh_x, gh_y = w.shape
    big_v[:gh_x, :gh_y] = w

    big_v = circshift(big_v, -floor(gh_x/2), -floor(gh_y/2))
    # big_v = circshift(big_v, -64, -64)

    # applying blur kernel on each channel in freq. domain
    y_f = np.zeros(x.shape, dtype=complex)
    for i in range(channels):
        y_f[:,:,i] = np.fft.fft2(x[:,:,i]) * np.fft.fft2(big_v)
        
    # converting to spatial domain
    y = np.zeros(x.shape)
    for i in range(channels):
        y[:,:,i] = np.fft.fftshift( np.fft.ifft2(y_f[:,:,i]).real)
    
    # adding noise
    noise = np.random.normal(0,noise_std, size = y.shape)
    y = y + noise
    y = np.clip(y,0,1)

    cv2.imwrite('./test_1.png', y)

    return y

def median_Blur_1(img, filiter_size = 3):  #当输入的图像为彩色图像
    image_copy = np.array(img, copy = True).astype(np.float32)
    processed = np.zeros_like(image_copy)
    middle = int(filiter_size / 2)
    r = np.zeros(filiter_size * filiter_size)
    g = np.zeros(filiter_size * filiter_size)
    b = np.zeros(filiter_size * filiter_size)
    
    for i in range(middle, image_copy.shape[1] - middle):
        for j in range(middle, image_copy.shape[2] - middle):
            count = 0
            #依次取出模板中对应的像素值
            for m in range(i - middle, i + middle +1):
                for n in range(j - middle, j + middle + 1):
                    r[count] = image_copy[0][m][n]
                    g[count] = image_copy[1][m][n]
                    b[count] = image_copy[2][m][n]
                    count += 1
            r.sort()
            g.sort()
            b.sort()
            processed[0][i][j] = r[int(filiter_size*filiter_size/2)]
            processed[1][i][j] = g[int(filiter_size*filiter_size/2)]
            processed[2][i][j] = b[int(filiter_size*filiter_size/2)]
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    processed = torch.from_numpy(processed)
    return processed

def median_Blur(img, filiter_size = 3):  #当输入的图像为彩色图像
    image_copy = np.array(img, copy = True).astype(np.float32)
    processed = np.zeros_like(image_copy)
    middle = int(filiter_size / 2)
    r = np.zeros(filiter_size * filiter_size)
    g = np.zeros(filiter_size * filiter_size)
    b = np.zeros(filiter_size * filiter_size)
    
    for i in range(middle, image_copy.shape[0] - middle):
        for j in range(middle, image_copy.shape[1] - middle):
            count = 0
            #依次取出模板中对应的像素值
            for m in range(i - middle, i + middle +1):
                for n in range(j - middle, j + middle + 1):
                    r[count] = image_copy[m][n][0]
                    g[count] = image_copy[m][n][1]
                    b[count] = image_copy[m][n][2]
                    count += 1
            r.sort()
            g.sort()
            b.sort()
            processed[i][j][0] = r[int(filiter_size*filiter_size/2)]
            processed[i][j][1] = g[int(filiter_size*filiter_size/2)]
            processed[i][j][2] = b[int(filiter_size*filiter_size/2)]
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    return processed


def data_normal(input):
    d_min = input.min()
    if d_min<0:
        input += torch.abs(d_min)
        d_min = input.min()
    d_max = input.max()
    dst = d_max - d_min
    output = (input - d_min).true_divide(dst)

    return output

def cacl_test(img1_iter, img2_iter):
    '''
    img1_iter: 模糊图像_batchsize 类型：numpy
    img2_iter: 清晰图像_batchsize 类型：numpy

    Return: Batchsize平均指标
    (PSNR_avg, SSIM_avg)
    '''

    num = img1_iter.shape[2]
    PSNRSum = 0
    SSIMSum = 0
    sumI = 0
    for iter in range(num):
        img1 = img1_iter[:, :, iter]
        img2 = img2_iter[:, :, iter]

        MSE = mean_squared_error(img1, img2)
        PSNR = peak_signal_noise_ratio(img1, img2)
        SSIM = structural_similarity(img1, img2, multichannel=True)

        PSNRSum += PSNR
        SSIMSum += SSIM
        sumI += 1

    PSNR_avg = PSNRSum/sumI
    SSIM_avg = SSIMSum/sumI

    return PSNR_avg, SSIM_avg

def NormMinandMax(npdarr, min=0, max=1):
    """"
    将数据npdarr 归一化到[min,max]区间的方法
    返回 副本
    """
    input = npdarr.copy()
    arr = input.flatten()
    Ymax = np.max(arr)  # 计算最大值
    Ymin = np.min(arr)  # 计算最小值
    k = (max - min) / (Ymax - Ymin)
    last = min + k * (npdarr - Ymin)

    return last

def complex_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first two dimensions.
    # Dimensions 3 and higher will have the same shape after multiplication.
    scalar_matmul = partial(torch.einsum, "ab..., cb... -> ac...")
 
 
    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet.  Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
    c = torch.zeros(real.shape, dtype=torch.complex64)
    c.real, c.imag = real, imag
 
 
    return c


def fft_conv_1d(
    signal: Tensor, kernel: Tensor, bias: Tensor = None, padding: int = 0,
) -> Tensor:
    """
    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.
    Returns:
        (Tensor) Convolved tensor
    """
    # 1. Pad the input signal & kernel tensors
    signal = f.pad(signal, [padding, padding])
    kernel_padding = [0, signal.size(-1) - kernel.size(-1)]
    padded_kernel = f.pad(kernel, kernel_padding)
 
 
    # 2. Perform fourier convolution
    signal_fr = rfftn(signal, dim=-1)
    kernel_fr = rfftn(padded_kernel, dim=-1)
 
 
    # 3. Multiply the transformed matrices
    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr)
 
 
    # 4. Compute inverse FFT, and remove extra padded values
    output = irfftn(output_fr, dim=-1)
    output = output[:, :, :signal.size(-1) - kernel.size(-1) + 1]
 
 
    # 5. Optionally, add a bias term before returning.
    if bias is not None:
        output += bias.view(1, -1, 1)
 
 
    return output

def blur_img_creat_self_1(sharp_input_cv, kernel_path):

    batch_size_test = sharp_input_cv.shape[0]
    sharp_input_cv_clone = sharp_input_cv.clone()
    sharp_input_cv_np = np.array(sharp_input_cv_clone/255, dtype=float)
    
    blur_output = torch.zeros(batch_size_test, 3, 128, 128)
    for j in range(batch_size_test):

        input = sharp_input_cv_np[j, :, :, :]
        
        # get kernel
        kernel_path_test = kernel_path[j]
        kernel_middle = scio.loadmat(kernel_path_test)
        kernel_tensor = kernel_middle['kernel']

        dim_test = kernel_tensor.ndim
        if dim_test>2:
            kernel_tensor = kernel_tensor.squeeze()

        # blurring
        # input= cv2.filter2D(input, -1, kernel_tensor)
        input = GenerateBlurry(input, kernel_tensor, 0.01)

        blur_output_middle = np.expand_dims(input, axis=0)
        if j == 0:
            blur_output = blur_output_middle
        else:
            blur_output = np.concatenate((blur_output, blur_output_middle), axis=0)


    # blur_output = np.array(blur_output/255, dtype=float)
    output = blur_output + 0.01 * np.float32(np.random.randn(*(blur_output.shape)))
    output = np.clip(output, 0, 1)
    output = np.uint8(blur_output*255)

    for ii in range(output.shape[0]):
        img_Image = Image.fromarray(cv2.cvtColor(output[ii],cv2.COLOR_BGR2RGB))  
        blur_output = transform(img_Image)
        blur_output_test = np.expand_dims(blur_output, axis=0)
        if ii == 0:
            blur_output_new = blur_output_test
        else:
            blur_output_new = np.concatenate((blur_output_new, blur_output_test), axis=0)

    blur_output_new = torch.from_numpy(blur_output_new)
    blur_output_new = blur_output_new.float()

    # blur_test = blur_output_new/2+0.5
    # torchvision.utils.save_image(blur_test, './1.png')

    return blur_output_new

def blurring_img(sharp_input, kernel_path):

    #generating blurry image
    batch_size_test = sharp_input.shape[0]
    input = sharp_input.clone()

    input = input.numpy()
    # input = (input/2+0.5) * 255

    # for j in range(batch_size_test):
    #     kernel_path_middle = kernel_path[j]
    #     # get kernel
    #     kernel_middle = scio.loadmat(kernel_path_middle)
    #     kernel_tensor = kernel_middle['kernel']
    #     kernels = np.array(kernel_tensor)
    #     kernels = kernels.transpose([1, 0])

    #     input[j,0,:,:]= signal.convolve(input[j,0,:,:],kernels[:,:],mode='same')
    #     input[j,1,:,:]= signal.convolve(input[j,1,:,:],kernels[:,:],mode='same')
    #     input[j,2,:,:]= signal.convolve(input[j,2,:,:],kernels[:,:],mode='same')

    for j in range(batch_size_test):
        # get kernel
        kernel_path_test = kernel_path[j]
        kernel_middle = scio.loadmat(kernel_path_test)
        kernel_tensor = kernel_middle['kernel']

        dim_test = kernel_tensor.ndim
        if dim_test>2:
            kernel_tensor = kernel_tensor.squeeze()

        # blurring
        # input[j,:,:,:]= cv2.filter2D(input[j,:,:,:], -1, kernel_tensor)
        input[j,0,:,:]= cv2.filter2D(input[j,0,:,:], -1, kernel_tensor)
        input[j,1,:,:]= cv2.filter2D(input[j,1,:,:], -1, kernel_tensor)
        input[j,2,:,:]= cv2.filter2D(input[j,2,:,:], -1, kernel_tensor)

    # input = input / 255
    # input = (input-0.5) * 2
    input = input + 0.01 * np.random.normal(0,1,input.shape)
    input = torch.from_numpy(input)
    input = input.float()

    torchvision.utils.save_image(input, 'test/test_img/normalize.png')
    a = input/2+0.5
    torchvision.utils.save_image(a, 'test/test_img/new.png')

    return input

def blur_img_creat_Blind_train(sharp_input_Image, kernel_path):

    batch_size_test = sharp_input_Image.shape[0]
    sharp_input_cv_cp = sharp_input_Image.clone()
    
    sharp_input_cv_np = np.array(sharp_input_cv_cp)
    input = sharp_input_cv_np.copy()

    for j in range(batch_size_test):
        # get kernel
        kernel_path_test = kernel_path[j]
        kernel_middle = scio.loadmat(kernel_path_test)
        kernel_tensor = kernel_middle['kernel']

        dim_test = kernel_tensor.ndim
        if dim_test>2:
            kernel_tensor = kernel_tensor.squeeze()

        kernel_np = np.array(kernel_tensor)
        kernel_np = kernel_np.transpose([1,0])

        # blurring
        input[j,0,:,:]= signal.convolve(input[j,0,:,:],kernel_np[:,:],mode='same')
        input[j,1,:,:]= signal.convolve(input[j,1,:,:],kernel_np[:,:],mode='same')
        input[j,2,:,:]= signal.convolve(input[j,2,:,:],kernel_np[:,:],mode='same')

    # 转换范围：[-1,1] --> [0,1]
    # 加噪
    input = input/2+0.5
    input = input + np.random.normal(0,0.01,size=input.shape)
    # clip范围
    input = np.clip(input, 0, 1)
    # 转换范围：[0,1] --> [-1,1]
    input = (input-0.5)*2
    # numpy --> torch_float
    input = torch.from_numpy(input).float()

    return input

def blur_img_creat_new(sharp_path, sharp_input_cv, kernel_path):

    batch_size_test = sharp_input_cv.shape[0]
    
    blur_output = torch.zeros(batch_size_test, 3, 128, 128)

    sharp_input_cv_new = np.array(sharp_input_cv/255, dtype=float)
    blur_img_ori_new = sharp_input_cv_new.copy()

    for j in range(batch_size_test):
        blurring = True
        input = blur_img_ori_new[j]
        input_path = sharp_path[j]

        dir_test = input_path.split('/')[-2]
        # if dir_test == 'sharp_7000':
        #     blurring = False
        #     blur_root = '/workspace/DataBase/train_dataset/Train/blur_7000'
        #     img_num = os.path.basename(input_path)

        #     blur_path = os.path.join(blur_root, img_num)
        #     blur_img_cv = cv2.imread(blur_path)
        #     input = np.array(blur_img_cv/255, dtype=float)

        # get kernel
        kernel_path_test = kernel_path[j]
        kernel_middle = scio.loadmat(kernel_path_test)
        kernel_tensor = kernel_middle['kernel']

        # blurring
        if blurring:
            input= cv2.filter2D(input, -1, kernel_tensor, borderType=cv2.BORDER_REPLICATE)
            input = input + 0.01 * np.float32(np.random.randn(*(input.shape)))

        blur_output_middle = np.expand_dims(input, axis=0)
        if j == 0:
            blur_output = blur_output_middle
        else:
            blur_output = np.concatenate((blur_output, blur_output_middle), axis=0)

    output = np.clip(blur_output, 0, 1)
    output = np.uint8(output*255)


    for ii in range(output.shape[0]):
        img_Image = Image.fromarray(cv2.cvtColor(output[ii],cv2.COLOR_BGR2RGB))  
        blur_output = transform(img_Image)
        blur_output_test = np.expand_dims(blur_output, axis=0)
        if ii == 0:
            blur_output_new = blur_output_test
        else:
            blur_output_new = np.concatenate((blur_output_new, blur_output_test), axis=0)


    blur_output_new = torch.from_numpy(blur_output_new)
    blur_output_new = blur_output_new.float()
    

    # blur_test = blur_output_new/2+0.5
    # torchvision.utils.save_image(blur_test, './1.png')

    return blur_output_new

def blur_img_creat_new_1(sharp_input_cv, kernel_path):

    batch_size_test = sharp_input_cv.shape[0]
    
    blur_output = torch.zeros(batch_size_test, 3, 128, 128)
    for j in range(batch_size_test):

        blur_img_ori = sharp_input_cv[j, :, :, :]
        input = blur_img_ori.clone()
        input = input.numpy()
        
        # get kernel
        kernel_path_test = kernel_path[j]
        kernel_middle = scio.loadmat(kernel_path_test)
        kernel_tensor = kernel_middle['kernel']

        dim_test = kernel_tensor.ndim
        if dim_test>2:
            kernel_tensor = kernel_tensor.squeeze()

        # blurring
        input= cv2.filter2D(input, -1, kernel_tensor, borderType=cv2.BORDER_REPLICATE)

        # input = GenerateBlurry(input, kernel_tensor, 0.01)
        # input = GenerateBlurry_self_1(input, kernel_tensor, 0.01)

        # signal = input
        # y1 = fft_conv_1d(signal, kernel_tensor, padding=128)


        blur_output_middle = np.expand_dims(input, axis=0)
        if j == 0:
            blur_output = blur_output_middle
        else:
            blur_output = np.concatenate((blur_output, blur_output_middle), axis=0)


    blur_output = np.array(blur_output/255, dtype=float)
    # blur_output = np.array(blur_output, dtype=float) # test21_4_3
    output = blur_output + 0.01 * np.float32(np.random.randn(*(blur_output.shape)))
    output = np.clip(output, 0, 1)
    output = np.uint8(output*255)

    for ii in range(output.shape[0]):
        img_Image = Image.fromarray(cv2.cvtColor(output[ii],cv2.COLOR_BGR2RGB))  
        blur_output = transform(img_Image)
        blur_output_test = np.expand_dims(blur_output, axis=0)
        if ii == 0:
            blur_output_new = blur_output_test
        else:
            blur_output_new = np.concatenate((blur_output_new, blur_output_test), axis=0)

    blur_output_new = torch.from_numpy(blur_output_new)
    blur_output_new = blur_output_new.float()

    # blur_test = blur_output_new/2+0.5
    # torchvision.utils.save_image(blur_test, './1.png')

    return blur_output_new

def blur_img_creat_new_2(sharp_path, sharp_input_cv, kernel_path):

    batch_size_test = sharp_input_cv.shape[0]
    
    blur_output = torch.zeros(batch_size_test, 3, 128, 128)

    sharp_input_cv_new = np.array(sharp_input_cv/255, dtype=float)
    blur_img_ori_new = sharp_input_cv_new.copy()

    for j in range(batch_size_test):
        blurring = True
        input = blur_img_ori_new[j]
        input_path = sharp_path[j]

        kernel_tensor = kernel_path[j]
        kernel_tensor = kernel_tensor.numpy()

        dir_test = input_path.split('/')[-2]
        if dir_test == 'sharp_7000':
            blurring = False
            blur_root = '/workspace/DataBase/train_dataset/Train/blur_7000'
            img_num = os.path.basename(input_path)

            blur_path = os.path.join(blur_root, img_num)
            blur_img_cv = cv2.imread(blur_path)
            input = np.array(blur_img_cv/255, dtype=float)

        # blurring
        if blurring:
            input= cv2.filter2D(input, -1, kernel_tensor)
            input = input + 0.01 * np.float32(np.random.randn(*(input.shape)))

        blur_output_middle = np.expand_dims(input, axis=0)
        if j == 0:
            blur_output = blur_output_middle
        else:
            blur_output = np.concatenate((blur_output, blur_output_middle), axis=0)

    output = np.clip(blur_output, 0, 1)
    output = np.uint8(output*255)


    for ii in range(output.shape[0]):
        img_Image = Image.fromarray(cv2.cvtColor(output[ii],cv2.COLOR_BGR2RGB))  
        blur_output = transform(img_Image)
        blur_output_test = np.expand_dims(blur_output, axis=0)
        if ii == 0:
            blur_output_new = blur_output_test
        else:
            blur_output_new = np.concatenate((blur_output_new, blur_output_test), axis=0)


    blur_output_new = torch.from_numpy(blur_output_new)
    blur_output_new = blur_output_new.float()
    

    # blur_test = blur_output_new/2+0.5
    # torchvision.utils.save_image(blur_test, './1.png')

    return blur_output_new

def blurring(sharp_path, input_cpu, kernel_path):
# def blurring(input_cpu, kernel_path):
    blurring = True
    batch_size,ch,x,y = input_cpu.size()
    x1 = int((x-128)/2)
    y1 = int((y-128)/2)

    #generating blurry image
    input_cpu = input_cpu.numpy()
    sharp_cpu = input_cpu.clone()
    kernels = kernel_path.numpy()
    for j in range(batch_size):
        input_path = sharp_path[j]
        dir_test = input_path.split('/')[-2]
        if dir_test == 'sharp_7000':
            blurring = False
            blur_root = '/workspace/DataBase/train_dataset/Train/blur_7000'
            img_num = os.path.basename(input_path)

            blur_path = os.path.join(blur_root, img_num)
            blur_img = Image.open(blur_path)
            input_cpu[j] = transform(blur_img).numpy()

        if blurring:
            input_cpu[j,0,:,:]= signal.convolve(input_cpu[j,0,:,:],kernels[j],mode='same')
            input_cpu[j,1,:,:]= signal.convolve(input_cpu[j,1,:,:],kernels[j],mode='same')
            input_cpu[j,2,:,:]= signal.convolve(input_cpu[j,2,:,:],kernels[j],mode='same')
    
    input_cpu = input_cpu + (1.0/255.0)* np.random.normal(0,4,input_cpu.shape)
    input_cpu = input_cpu[:,:,x1:x1+128,y1:y1+128]
    sharp_cpu = sharp_cpu[:,:,x1:x1+128,y1:y1+128]
    
    input_cpu = torch.from_numpy(input_cpu)
    sharp_cpu = torch.from_numpy(sharp_cpu)
    # resize 175 --> 128
    # resize_test = transforms.Resize((128, 128)) 
    # input_cpu = resize_test(input_cpu)

    input_cpu = input_cpu.float().cuda()
    sharp_cpu = sharp_cpu.float().cuda()

    return input_cpu, sharp_cpu




def openDir(bluringImgPath, GTPath):
    bluringList = []
    GTList = []
    for root, dirs, files in os.walk(bluringImgPath):  
        bluringList=files
    for root, dirs, files in os.walk(GTPath):
        GTList=files
    return bluringList,GTList

def metrics(blur_img, true_img):
    bluringImgPath = blur_img
    GTPath = true_img
    bList,GList = openDir(bluringImgPath, GTPath)

    if bList.sort() != GList.sort():
        print("Image name Error")
    PSNRSum = 0
    SSIMSum = 0
    sumI = 0
    for filename in bList:
        BP = bluringImgPath+"/"+filename
        GP = GTPath+"/"+filename

        img1 = cv2.imread(BP)
        img2 = cv2.imread(GP)

        MSE = mean_squared_error(img1, img2)
        PSNR = peak_signal_noise_ratio(img1, img2)
        SSIM = structural_similarity(img1, img2, multichannel=True)

        PSNRSum += PSNR
        SSIMSum += SSIM
        sumI += 1
    
    PSNR_avg = PSNRSum/sumI
    SSIM_avg = SSIMSum/sumI

    return PSNR_avg, SSIM_avg

def metrics_3(blur_img, true_img):
    result_img_path_ori = os.path.join('/workspace/DataBase/Valiation_all/Ijcv_test_1000/result_new', 'cli_test/')
    os.makedirs(result_img_path_ori, exist_ok=True)
    bluringImgPath = blur_img
    GTPath = true_img
    bList,GList = openDir(bluringImgPath, GTPath)

    if bList.sort() != GList.sort():
        print("Image name Error")

    PSNRSum = 0
    SSIMSum = 0
    sumI = 0

    for filename in bList:
        set_num = 3

        BP = bluringImgPath+"/"+filename
        GP = GTPath+"/"+filename

        img1 = cv2.imread(BP)
        img2 = cv2.imread(GP)

        PSNR_scoring_max = 0
        SSIM_scoring_max = 0

        num_test = img1.shape[0]

        PSNR_left_new = peak_signal_noise_ratio(img1, img2)
        SSIM_left_new = structural_similarity(img1, img2, multichannel=True)

        PSNR_scoring_ori = PSNR_left_new
        SSIM_scoring_ori = SSIM_left_new

        if PSNR_left_new > PSNR_scoring_max:
            PSNR_scoring_max = PSNR_left_new

        if SSIM_left_new > SSIM_scoring_max:
                SSIM_scoring_max = SSIM_left_new

        # 上移
        for j in range(set_num):
            
            img1_new = img1[j+1:, :, :]
            img2_new = img2[:num_test-j-1, :, :]

            PSNR_left_new = peak_signal_noise_ratio(img1_new, img2_new)
            SSIM_left_new = structural_similarity(img1_new, img2_new, multichannel=True)

            if PSNR_left_new > PSNR_scoring_max:
                PSNR_scoring_max = PSNR_left_new

            if SSIM_left_new > SSIM_scoring_max:
                SSIM_scoring_max = SSIM_left_new

        # 下移
        for j in range(set_num):
            img1_new = img1[:num_test-j-1, :, :]
            img2_new = img2[j+1:, :, :]

            PSNR_left_new = peak_signal_noise_ratio(img1_new, img2_new)
            SSIM_left_new = structural_similarity(img1_new, img2_new, multichannel=True)

            if PSNR_left_new > PSNR_scoring_max:
                PSNR_scoring_max = PSNR_left_new

            if SSIM_left_new > SSIM_scoring_max:
                SSIM_scoring_max = SSIM_left_new

        # 左移
        for j in range(set_num):
            img1_new = img1[:, j+1:, :]
            img2_new = img2[:, :num_test-j-1, :]

            PSNR_left_new = peak_signal_noise_ratio(img1_new, img2_new)
            SSIM_left_new = structural_similarity(img1_new, img2_new, multichannel=True)

            if PSNR_left_new > PSNR_scoring_max:
                PSNR_scoring_max = PSNR_left_new

            if SSIM_left_new > SSIM_scoring_max:
                SSIM_scoring_max = SSIM_left_new

        # 右移
        for j in range(set_num):
            img1_new = img1[:, :num_test-j-1, :]
            img2_new = img2[:, j+1:, :]

            PSNR_left_new = peak_signal_noise_ratio(img1_new, img2_new)
            SSIM_left_new = structural_similarity(img1_new, img2_new, multichannel=True)

            if PSNR_left_new > PSNR_scoring_max:
                PSNR_scoring_max = PSNR_left_new

            if SSIM_left_new > SSIM_scoring_max:
                SSIM_scoring_max = SSIM_left_new

        # 左上移
        for i in range(set_num):
            for j in range(set_num):
                img1_new = img1[i+1:, j+1:, :]
                img2_new = img2[:num_test-i-1, :num_test-j-1, :]

                PSNR_left_new = peak_signal_noise_ratio(img1_new, img2_new)
                SSIM_left_new = structural_similarity(img1_new, img2_new, multichannel=True)

                if PSNR_left_new > PSNR_scoring_max:
                    PSNR_scoring_max = PSNR_left_new

                if SSIM_left_new > SSIM_scoring_max:
                    SSIM_scoring_max = SSIM_left_new

        # 左下移
        for i in range(set_num):
            for j in range(set_num):
                img1_new = img1[:num_test-i-1, j+1:, :]
                img2_new = img2[i+1:, :num_test-j-1, :]

                PSNR_left_new = peak_signal_noise_ratio(img1_new, img2_new)
                SSIM_left_new = structural_similarity(img1_new, img2_new, multichannel=True)

                if PSNR_left_new > PSNR_scoring_max:
                    PSNR_scoring_max = PSNR_left_new

                if SSIM_left_new > SSIM_scoring_max:
                    SSIM_scoring_max = SSIM_left_new

        # 右上移
        for i in range(set_num):
            for j in range(set_num):
                img1_new = img1[i+1:, :num_test-j-1, :]
                img2_new = img2[:num_test-i-1, j+1:, :]

                PSNR_left_new = peak_signal_noise_ratio(img1_new, img2_new)
                SSIM_left_new = structural_similarity(img1_new, img2_new, multichannel=True)

                if PSNR_left_new > PSNR_scoring_max:
                    PSNR_scoring_max = PSNR_left_new

                if SSIM_left_new > SSIM_scoring_max:
                    SSIM_scoring_max = SSIM_left_new

        # 右下移
        for i in range(set_num):
            for j in range(set_num):
                img1_new = img1[:num_test-i-1, :num_test-j-1, :]
                img2_new = img2[i+1:, j+1:, :]

                PSNR_left_new = peak_signal_noise_ratio(img1_new, img2_new)
                SSIM_left_new = structural_similarity(img1_new, img2_new, multichannel=True)

                if PSNR_left_new > PSNR_scoring_max:
                    PSNR_scoring_max = PSNR_left_new

                if SSIM_left_new > SSIM_scoring_max:
                    SSIM_scoring_max = SSIM_left_new

        PSNRSum += PSNR_scoring_max
        SSIMSum += SSIM_scoring_max
        
        PSNR_up = float(PSNR_scoring_max - PSNR_scoring_ori)
        SSIM_up = float(SSIM_scoring_max - SSIM_scoring_ori)
        print(PSNR_up)
        print(SSIM_up)
        print('---------------')
        sumI += 1
    
    PSNR_avg = PSNRSum/sumI
    SSIM_avg = SSIMSum/sumI

    return PSNR_avg, SSIM_avg


def mk_dir(path):
    os.makedirs(path, exist_ok=True)

# 计算当前步数下，初始化位置坐标的平均学习率
def Avg_lr(lr_reals, lr_dot, stride=1):
    # lr_reals：所有参数当前步数下的，学习率
    # lr_dot: 预设置的“用于计算avg_Lr”的位置坐标(每一层除偏置层取100个参数)
    sample_tensor = []
    for i in range(0, len(lr_reals), stride):
        sample_tensor.append(lr_reals[i])

    sum = 0
    count = 0
    for i in range(len(sample_tensor)):
        x = sample_tensor[i]
        y_50 = lr_dot[i]

        for j in range(len(y_50)):
            y = y_50[j]
            
            y_len = len(y)
            if y_len == 1:
                dot = x[y[0]]
                # print(dot)
                sum += dot
                count += 1
            elif y_len == 4:
                print(y_len)
                print(y.shape)
                dot = x[y[0], y[1], y[2], y[3]]
                sum += dot
                count += 1    
            else:
                dot = x[y[0], y[1]]
                # print(dot)
                sum += dot
                count += 1  
    
    avg_result = (sum / count)
    return avg_result

# 保存"D_lr"初始位置信息
def lr_sample(lr_reals, save_path='/models', stride=1):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lr_reals_len = len(lr_reals)
    sample_lr = []
    sample_num = 100
    for i in range(0, lr_reals_len, stride):
        b = lr_reals[i]
        c = torch.rand([sample_num,len(b.shape)])

        b_shape = torch.tensor(b.shape)
        d = c * b_shape
        
        d = d.floor()
        d = d.int()

        sample_lr.append(d)
    # print(len(sample_lr))
    torch.save(sample_lr, save_path)

# adam_init
def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    lr_reals = []

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1
        # lr_t 
        # step_size_1 = step_size * math.sqrt(bias_correction2)
        # 自己计算得到的学习率
        lr_real = (step_size * (1 - beta1)) / denom
        lr_reals.append(lr_real)

        param.addcdiv_(exp_avg, denom, value=-step_size)
    lr_reals_output = lr_reals
    # for i in range(len(lr_reals)):
    #     print("第{0}层：{1}".format(i, lr_reals[i].shape))
    return lr_reals_output

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        unloader = transforms.ToPILImage()
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

def writeToFile(varname='None', varvalue='  ', filename='logs/log', clear=False, num=None):
    if clear:
        return rm_writeToFile(filename)

    with open(filename, 'a') as log:
        if num:
            log.write('{}epoch: {}:\n\t{}\n'.format(num,varname, varvalue)),
            log.write('--------------------------------------------------------------------------------------\n')
        else:
            log.write('{}:\n\t{}\n'.format(varname, varvalue)),
            log.write('--------------------------------------------------------------------------------------\n')

def rm_writeToFile(filename="./logs/log"):
    if not os.path.exists(filename):
        return
    os.remove(filename)

def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def default(value, d):
    return value if exists(value) else d

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    return el if isinstance(el, list) else [el]

def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts =  head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def image_noise(images, device, model, n=16):
    noise_vgg = model(images)
    for i in range(len(noise_vgg)):
        noise_vgg[i] = noise_vgg[i].cuda(device)

    noise_vgg.reverse()

    return noise_vgg

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

# losses

def gen_hinge_loss(fake, real):
    return fake.mean()

def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

# dataset

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

def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

# dataset transform
def transform(img):

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        # transforms.RandomCrop((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    x = transform(img)

    return x

def transform_d(img):

    transform = transforms.Compose([
        # transforms.Resize((128,128)),
        transforms.RandomCrop((128,128)),
        # transforms.CenterCrop((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    x = transform(img)

    return x

def transform_g(img):

    transform = transforms.Compose([
        transforms.RandomCrop((128,128)),
        # transforms.CenterCrop((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    x = transform(img)

    return x

def transform_1(img):

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    x_np = np.array(img)
    x_np = im2double(x_np)
    x_np = np.array(x_np, dtype=float)
    x_np = np.uint8(x_np*255)
    x_np = Image.fromarray(x_np.astype('uint8')).convert('RGB')
    x = transform(x_np)

    return x

def transform_2(img):

    transform = transforms.Compose([
        # transforms.Resize((175,175)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    x = transform(img)

    return x

def loadZ(path, path1, count=32):
    global img_files
    kk =0 
    img_files = []
    fname_paths = []
    fname_paths_1 = []
    fname = []
    fname1=[]

    if fname in sorted(os.listdir(path)):
        for fname in sorted(os.listdir(path)):  
            init_path = os.path.join(path, fname)
            fname_path = recursive_listdir(init_path)
    else:
        fname_path = recursive_listdir(path)

    img_files = []

    for kk in range(count):
        fname_paths.append(fname_path[kk])
        
        img_num1 = fname_path[kk].split('/')[-2]
        img_num2 = fname_path[kk].split('/')[-1]

        img = os.path.join(path1, img_num1, img_num2)
        fname_paths_1.append(img)

        kk += 1

    vgg_img = torch.zeros(count, 3, 128, 128)
    D_img = torch.zeros(count, 3, 128, 128)

    for i in range(count):
        vgg_img_image = Image.open(os.path.join(path, fname_paths[i]))
        D_img_image = Image.open(os.path.join(path1, fname_paths_1[i]))
        vgg_img_image = transform(vgg_img_image)
        D_img_image = transform(D_img_image)
        
        vgg_img[i, :, :, :] = vgg_img_image
        D_img[i, :, :, :] = D_img_image

    return vgg_img, D_img

def loadZ_val(path, path1, count=32):
    global img_files
    kk =0 
    img_files = []
    fname_paths = []
    fname_paths_1 = []
    fname = []
    fname1=[]

    if fname in sorted(os.listdir(path)):
        for fname in sorted(os.listdir(path)):  
            init_path = os.path.join(path, fname)
            fname_path = recursive_listdir(init_path)
    else:
        fname_path = recursive_listdir(path)

    img_files = []
    count = len(fname_path)

    for kk in range(count):
        fname_paths.append(fname_path[kk])

        img_num1 = fname_path[kk].split('/')[-2]
        img_num2 = fname_path[kk].split('/')[-1]
        # num_test = img_num2.split('_')[0]
        # img = os.path.join(path1, num_test+'.png')
        img = os.path.join(path1, img_num1, img_num2)
        
        fname_paths_1.append(img)

        kk += 1

    vgg_img = torch.zeros(count, 3, 128, 128)
    D_img = torch.zeros(count, 3, 128, 128)

    for i in range(count):
        vgg_img_image = Image.open(os.path.join(path, fname_paths[i]))
        D_img_image = Image.open(os.path.join(path1, fname_paths_1[i]))
        vgg_img_image = transform(vgg_img_image)
        D_img_image = transform(D_img_image)
        
        vgg_img[i, :, :, :] = vgg_img_image
        D_img[i, :, :, :] = D_img_image

    return vgg_img, D_img, fname_paths, fname_paths_1

class loadz():
    def __init__(self, path, path1, count=32):
        self.path = path
        self.path1 = path1
        self.count = count
    
    def loadz_main(self, file_get='file', evaluate_img=True):
        global img_files
        kk =0 
        img_files = []
        fname_paths = []
        fname_paths_1 = []
        fname = []

        if fname in sorted(os.listdir(self.path)):
            for fname in sorted(os.listdir(self.path)):  
                init_path = os.path.join(self.path, fname)
                fname_path = recursive_listdir(init_path)
        else:
            fname_path = recursive_listdir(self.path)

        img_files = []

        if evaluate_img:
            count = self.count
        else:
            count = len(fname_path)

        for kk in range(count):
            fname_paths.append(fname_path[kk])

            img_num1 = fname_path[kk].split('/')[-1]
            img_num2 = fname_path[kk].split('/')[-2]
            img_num3 = fname_path[kk].split('/')[-3]

            if file_get == 'file':
                img = os.path.join(self.path1, img_num1)
            elif file_get == 'dir':
                img = os.path.join(self.path1, img_num2, img_num1)
            elif file_get == 'dir_2':
                img = os.path.join(self.path1, img_num3, img_num2, img_num1)

            fname_paths_1.append(img)
            kk += 1

        vgg_img = torch.zeros(count, 3, 128, 128)
        D_img = torch.zeros(count, 3, 128, 128)

        for i in range(count):
            vgg_img_image = Image.open(fname_paths[i]).convert("RGB")
            D_img_image = Image.open(fname_paths_1[i]).convert("RGB")
            vgg_img_image = transform(vgg_img_image)
            D_img_image = transform(D_img_image)
            
            vgg_img[i, :, :, :] = vgg_img_image
            D_img[i, :, :, :] = D_img_image

        return vgg_img, D_img, fname_paths, fname_paths_1

# augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


if __name__ == '__main__':
    # image_path = './test.png'
    image_path = 'test/test_img/test_new.png'
    kernel_path = './10.mat'
    blur_save_img = './test_blur.png'

    image = Image.open(image_path)
    image = image.convert("RGB")
    resize = transforms.Resize((128, 128))
    a = transforms.ToTensor()

    image_train = resize(image)
    image_train = a(image_train)
    image_train = torch.unsqueeze(image_train, dim=0).cuda(0)

    # sharp_img_cv = cv2.imread(image_path)
    # sharp_img_cv = cv2.resize(sharp_img_cv, (128, 128))
    # sharp_img_cv = np.expand_dims(sharp_img_cv, axis=0)
    # blur_img = blur_img_creat_test(sharp_img_cv, kernel_path)

    noise_vgg = model_vgg(image_train)

    print(noise_vgg)
    for i in range (len(noise_vgg)):
        save_tensor_ori = noise_vgg[i]
        save_tensor_middle = save_tensor_ori[0, 0, :, :]
        save_path = 'test/test_img/1-4/'+str(i)+'.png'
        print(save_path)

        torchvision.utils.save_image(save_tensor_middle, save_path)