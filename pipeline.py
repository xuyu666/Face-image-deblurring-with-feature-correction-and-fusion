from datetime import datetime
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
import scipy.io as scio

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

from model_vgg import FeatureExtraction
from model import *
from help_class import *
from help_func import *

# constants
NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png', 'JPEG']
img_files = []
table = True
global_a = True
rm_writer_file = True
lr_reals = []


class Trainer():
    def __init__(
        self,
        writer = None,
        rank_init = 0,
        data = './data',
        runs_dir = ' ',
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        image_size = 128,
        network_capacity = 16,
        fmap_max = 512,
        transparent = False,
        batch_size = 4,
        mixed_prob = 0.9,
        gradient_accumulate_every=1,
        lr = 2e-4,
        lr_mlp = 0.1,
        ttur_mult = 2,
        rel_disc_loss = False,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 1000,
        num_image_tiles = 8,
        trunc_psi = 0.6,
        fp16 = False,
        cl_reg = False,
        no_pl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        aug_prob = 0.,
        aug_types = ['translation', 'cutout'],
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        dual_contrast_loss = False,
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        log = False,
        *args,
        **kwargs
    ):
        self.writer = writer
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.epoch = 0
        # self.shuffle_epoch = 0
        self.re_train = True
        self.name = name
        self.global_a = True
        self.global_b = True
        self.data = data
        
        self.runs_dir = runs_dir
        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        # self.models_dir = Path('/data/Experiment_models')
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.models_dir / name / 'config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.num_image_tiles = num_image_tiles
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0
        
        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.content_loss = 0
        cacul_content_loss = 0

        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.kernel_loader = None
        self.loader_D = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        self.dual_contrast_loss = dual_contrast_loss

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.is_ddp = is_ddp
        self.is_main = rank == rank_init
        self.rank = rank
        self.world_size = world_size

        self.logger = aim.Session(experiment=name) if log else None

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}
        
    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr = self.lr, lr_mlp = self.lr_mlp, ttur_mult = self.ttur_mult, image_size = self.image_size, network_capacity = self.network_capacity, fmap_max = self.fmap_max, transparent = self.transparent, fq_layers = self.fq_layers, fq_dict_size = self.fq_dict_size, attn_layers = self.attn_layers, fp16 = self.fp16, cl_reg = self.cl_reg, no_const = self.no_const, rank = self.rank, *args, **kwargs)
        model_vgg = FeatureExtraction().cuda(self.rank)

        if not self.is_ddp:
            self.model_vgg = model_vgg
        # self.model_vgg = FeatureExtraction().cuda(self.rank)
        if self.is_ddp:
            # model_vgg = FeatureExtraction().cuda(self.rank)
            ddp_kwargs = {'device_ids': [self.rank]}
            self.model_vgg = DDP(model_vgg, **ddp_kwargs, find_unused_parameters=True)
            # self.S_ddp = DDP(self.GAN.S, **ddp_kwargs, find_unused_parameters=True)
            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs, find_unused_parameters=True)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs, find_unused_parameters=True)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs, find_unused_parameters=True)

        if exists(self.logger):
            self.logger.set_params(self.hparams)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.fmap_max = config.pop('fmap_max', 512)
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp, 'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, kernel_folder, folder):
        # num_workers = num_workers = default(self.num_workers, NUM_CORES if not self.is_ddp else 0)
        num_workers = 8
        # MyDataset函数：VGG_train
        # self.dataset = MyDataset(os.path.join(folder, 'blur_7000'), os.path.join(folder, 'sharp_7000'))
        # self.dataset = MyDataset_new(os.path.join(folder, 'sharp1'))

        self.dataset = G_Dataset(os.path.join(folder, 'sharp1'), kernel_folder)
        self.sampler_g = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = self.sampler_g, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)
        # self.loader = iter(dataloader)

        # self.kernel_dataset = MyDataset_kernel(kernel_folder)  
        # kernel_sampler = DistributedSampler(self.kernel_dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        # kernel_dataloader = data.DataLoader(self.kernel_dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = kernel_sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        # self.kernel_loader = cycle(kernel_dataloader)

        # Dataset函数：D_train(原版数据集处理 )
        self.dataset_D = D_Dataset(os.path.join(folder, 'sharp2'), self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        self.sampler_d = DistributedSampler(self.dataset_D, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader_D = data.DataLoader(self.dataset_D, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = self.sampler_d, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader_D = cycle(dataloader_D)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(self.loader),  'You must first initialize the data source with noise'
        assert exists(self.loader_D), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_loss = torch.tensor(0.).cuda(self.rank)
        total_content_loss = torch.tensor(0.).cuda(self.rank)
        total_Labal_loss = torch.tensor(0.).cuda(self.rank)
        # contentLoss = PerceptualLoss(nn.MSELoss())
        contentLoss = nn.MSELoss()

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob   = self.aug_prob
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}
        
        apply_gradient_penalty = self.steps % 4 == 0
        apply_contentLoss = True
        # apply_path_penalty = not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0
        apply_path_penalty = False
        apply_cl_reg_to_generated = self.steps > 20000

        # S = self.GAN.S if not self.is_ddp else self.S_ddp
        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        backwards = partial(loss_backwards, self.fp16)

        # D_cl 中使用的生成图像方式为初始版本。
        if exists(self.GAN.D_cl):
            self.GAN.D_opt.zero_grad()

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    # get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
                    # style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
                    vgg_img = next(self.loader)
                    vgg_img = vgg_img.cuda(self.rank)
                    noise = image_noise(vgg_img, device=self.rank)
                    for i in range(len(noise)):
                        noise[i].cuda(self.rank)

                    w_space = latent_to_w(self.GAN.S, style)
                    w_styles = styles_def_to_tensor(w_space)

                    generated_images = self.GAN.G(vgg_img, device=self.rank)
                    self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                image_batch = next(self.loader_D).cuda(self.rank)
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.clone().detach().item()
            backwards(loss, self.GAN.D_opt, loss_id = 0)

            self.GAN.D_opt.step()

        # setup losses

        if not self.dual_contrast_loss:
            D_loss_fn = hinge_loss
            G_loss_fn = gen_hinge_loss
            G_requires_reals = False
        else:
            D_loss_fn = dual_contrastive_loss
            G_loss_fn = dual_contrastive_loss
            G_requires_reals = True

    # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, G]):

            # shuffle_epoch_now = floor(((self.steps+1) * (self.batch_size * 2) * self.gradient_accumulate_every) / len(self.dataset))
            # if shuffle_epoch_now > self.shuffle_epoch:
            #     self.shuffle_epoch = shuffle_epoch_now
            #     # 注意代码顺序
            #     self.sampler_d.set_epoch(self.shuffle_epoch+1)
            #     self.sampler_g.set_epoch(self.shuffle_epoch+1)

            #     filename = './model_log/shuffling.txt'
            #     with open (filename,'a') as file_object:
            #         file_object.write('shuffle_epoch: {}\n'.format(self.shuffle_epoch+1)) 

            vgg_img_d, _ = next(self.loader)
            vgg_img = vgg_img_d.cuda(self.rank)
            # torchvision.utils.save_image(vgg_img/2+0.5, './demo/sharp_g.png')
            
            vgg_feature = help_func.image_noise(vgg_img, device=self.rank, model=self.model_vgg)

            # 传入Image打开图像

            generated_images = G(vgg_img, vgg_feature, device=self.rank)
            fake_output, fake_q_loss = D_aug(generated_images.clone().detach(), detach = True, **aug_kwargs)

            image_batch = next(self.loader_D)
            # torchvision.utils.save_image(image_batch/2+0.5, './demo/sharp_d.png')

            image_batch = image_batch.cuda(self.rank)
            image_batch.requires_grad_()
            real_output, real_q_loss = D_aug(image_batch, **aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output

            if self.rel_disc_loss:
                real_output_loss = real_output_loss - fake_output.mean()
                fake_output_loss = fake_output_loss - real_output.mean()

            # 是否选择 dual_contrast_loss
            divergence = D_loss_fn(real_output_loss, fake_output_loss)
            disc_loss = divergence   

            if self.has_fq:
                quantize_loss = (fake_q_loss + real_q_loss).mean()
                self.q_loss = float(quantize_loss.detach().item())

                disc_loss = disc_loss + quantize_loss

            #   每四步实行梯度惩罚
            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, loss_id = 1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)

        self.GAN.D_opt.step()

    # train generator
        self.GAN.G_opt.zero_grad()
        cacul_content_loss = 0

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[G, D_aug]):

            vgg_img_g, sharp_img_g = next(self.loader)
            vgg_img = vgg_img_g.cuda(self.rank)
            sharp_img_g = sharp_img_g.cuda(self.rank)
            
            vgg_feature = help_func.image_noise(vgg_img, device=self.rank, model=self.model_vgg)

            generated_images = G(vgg_img, vgg_feature, device=self.rank)
            fake_output, _ = D_aug(generated_images, **aug_kwargs)
            fake_output_loss = fake_output

            real_output = None
            if G_requires_reals:
                image_batch = next(self.loader_D).cuda(self.rank)
                real_output, _ = D_aug(image_batch, detach = True, **aug_kwargs)
                real_output = real_output.detach()

            if self.top_k_training:
                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                # 向上取整
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

            # 是否需要 dual_contrastive_loss（G_requires_reals）
            loss = G_loss_fn(fake_output_loss, real_output)
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss
                  
        # ----------------------------------  content_Loss  ---------------------------------------
            #region
            '''例子：content_loss = contentLoss.get_loss(self.fake_B, self.real_B) * self.opt.lambda_A'''
            if apply_contentLoss:
                G_img_score = image_noise(generated_images, device=self.rank, model=self.model_vgg)
                Sharp_img_score = image_noise(sharp_img_g, device=self.rank, model=self.model_vgg)

                content_loss = contentLoss(G_img_score[8], Sharp_img_score[8])

                cacul_content_loss = content_loss * 100
                Labal_loss = nn.MSELoss()(generated_images, sharp_img_g) * 1000
                gen_loss = gen_loss + cacul_content_loss  + Labal_loss
                gen_loss = gen_loss + Labal_loss
                
            # endregion

            gen_loss = gen_loss / (self.gradient_accumulate_every)
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt, loss_id = 2)

            total_gen_loss += loss.detach().item() / (self.gradient_accumulate_every)
            total_content_loss += cacul_content_loss.detach().item() / (self.gradient_accumulate_every)
            total_Labal_loss += Labal_loss.detach().item() / (self.gradient_accumulate_every)

        self.g_loss = float(total_gen_loss)
        self.content_loss = float(total_content_loss)
        self.Labal_loss = float(total_Labal_loss)

        self.GAN.G_opt.step()

        # calculate moving averages
        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')
            self.writer_runs(self.pl_mean, 'PL')
            
        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results
        if self.is_main:
            # 注意点_batch_size: self.batch_size / ddp_size(每块使用的"GPU"分到的Batch_size)
            epoch_now = floor((self.steps * self.batch_size * self.gradient_accumulate_every) / len(self.dataset))
            # Ijcv 测试集_1000
            blur_path_valiation = '/workspace/DataBase/Compare_dataset/Test_data_Celea/blur_1000'
            sharp_path_valiation = '/workspace/DataBase/Compare_dataset/Test_data_Celea/sharp_1000'
            py_name = os.path.basename(__file__).split(".")[0]
            result_img_path_ori = os.path.join('/workspace/DataBase/Compare_dataset/Test_data_Celea/result', py_name, str(self.epoch))
            GTPath = sharp_path_valiation
            val_batch_size_ori = 20

            mk_dir(path=blur_path_valiation)
            mk_dir(path=sharp_path_valiation)
            if os.path.exists(result_img_path_ori):
                shutil.rmtree(path=result_img_path_ori)
            mk_dir(path=result_img_path_ori)
            

            if epoch_now > self.epoch: 
                self.epoch = epoch_now
                self.save(self.epoch, epoch_save=True)

                if self.epoch < 16 and not self.re_train:
                    self.GAN.G_scheduler.step()
                    self.GAN.D_scheduler.step()

                # 余弦学习率衰减：2e-4 --> 2e-5(10 epochs迭代)
                # 余弦学习率衰减：2e-4 --> 1e-5(20 epochs迭代)
                if self.re_train:
                    for i in range(self.epoch):
                        self.GAN.G_scheduler.step()
                        self.GAN.D_scheduler.step()
                    self.re_train = False

                filename = './model_log/programing.txt'
                for param_group in self.GAN.G_opt.param_groups:
                    with open (filename,'a') as file_object:
                        file_object.write('{}epoch_G_lr: {}\n'.format(self.epoch, param_group['lr'])) 

                for param_group in self.GAN.D_opt.param_groups:
                    with open (filename,'a') as file_object:
                        file_object.write('{}epoch_D_lr: {}\n'.format(self.epoch, param_group['lr']))  


                # --------生成验证集---------
                self.valiation(blur_path_valiation, sharp_path_valiation, floor(self.steps / self.evaluate_every), result_img_path=result_img_path_ori, val_batch_size=val_batch_size_ori, valiation=True)
                
                bluringImgPath = result_img_path_ori
                PSNR_avg, SSIM_avg = metrics(bluringImgPath, GTPath)
                print('{}_epoch_PSNR:{}'.format(self.epoch, PSNR_avg))
                print('{}_epoch_SSIM:{}'.format(self.epoch, SSIM_avg))
                # --------生成验证集---------

                self.writer_runs('Epoch_loss/loss_g',self.g_loss, self.epoch)
                self.writer_runs('Epoch_loss/loss_d',self.d_loss, self.epoch)
                self.writer_runs('Epoch_other_loss/Content_loss',self.content_loss,self.epoch)
                self.writer_runs('Epoch_other_loss/Labal_loss',self.Labal_loss,self.epoch)
                self.writer_runs('Mertric/PSNR',PSNR_avg,self.epoch)
                self.writer_runs('Mertric/SSIM',SSIM_avg,self.epoch)             

            # if self.steps >= 0 :  
            if self.steps % 10 == 0:
                self.writer_runs('loss/loss_g',self.g_loss,self.steps)
                self.writer_runs('loss/loss_d',self.d_loss,self.steps)
                self.writer_runs('other_loss/Content_loss',self.content_loss,self.steps)
                self.writer_runs('other_loss/Labal_loss',self.Labal_loss,self.steps)

            # 每1000 steps保存一次模型
            if self.steps % (self.save_every) == 0:
                self.save(self.checkpoint_num)

            # if self.steps % 10 == 0:
            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(blur_path_valiation, sharp_path_valiation, floor(self.steps / self.evaluate_every), valiation=True, eval_add=True)
  
        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, blur_path, sharp_path, num = 0, file_get_input='file', trunc = 1.0, valiation=False, eval_add=False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = self.num_image_tiles

        noise_normalize_img_path = os.path.join(self.results_dir / self.name / f'noise_normalize.jpg')
        noise_img_path = os.path.join(self.results_dir / self.name / f'noise.jpg')
        sharp_img_path = os.path.join(self.results_dir / self.name / f'sharp.jpg')
        result_img_path = os.path.join(self.results_dir / self.name / f'{str(num)}.{ext}')

        if valiation:
            self.global_a = True
            valiation_root = os.path.join(self.results_dir / self.name /'valiation')
            if not os.path.exists(valiation_root):
                os.makedirs(valiation_root,exist_ok=True)
            noise_normalize_img_path = os.path.join(valiation_root, f'noise_normalize.jpg')
            noise_img_path = os.path.join(valiation_root, f'noise.jpg')
            sharp_img_path = os.path.join(valiation_root, f'sharp.jpg')
            result_img_path = os.path.join(valiation_root, f'{str(num)}.{ext}')

        # ---------------------------------------
        Blur_path = blur_path
        Sharp_path = sharp_path
        # vgg_img_eval,sharp_img_eval = loadZ(Blur_path, Sharp_path,num_rows ** 2)

        loadz_test = loadz(Blur_path, Sharp_path, num_rows ** 2)
        vgg_img_eval,sharp_img_eval,_,_ = loadz_test.loadz_main(file_get=file_get_input)
        # ---------------------------------------

        # torchvision.utils.save_image(vgg_img, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        if self.global_a :
            torchvision.utils.save_image(vgg_img_eval, noise_normalize_img_path)
            vgg_img = vgg_img_eval/2 + 0.5
            torchvision.utils.save_image(vgg_img, noise_img_path)
            sharp_img = sharp_img_eval/2 + 0.5
            torchvision.utils.save_image(sharp_img, sharp_img_path)
            self.global_a = False

        vgg_img_eval = vgg_img_eval.cuda(self.rank)
        vgg_feature = help_func.image_noise(vgg_img_eval, device=self.rank, model=self.model_vgg)

        self.writer.flush()

        # regular
        generated_images = self.GAN.G(vgg_img_eval, vgg_feature, device=self.rank)
        # 数据集"Normalization(0.5, 0.5)"初始化, 归一化数据范围：[-1, 1]
        generated_images = generated_images/2 + 0.5

        torchvision.utils.save_image(generated_images, result_img_path)

    @torch.no_grad()
    def valiation(self, blur_path, sharp_path, num = 0, file_get_test='file', result_img_path=None, val_batch_size=1, trunc = 1.0, valiation=False):
        self.GAN.eval()
        num_rows = self.num_image_tiles

        # ---------------------------------------
        Blur_path = blur_path
        Sharp_path = sharp_path

        loadz_test = loadz(Blur_path, Sharp_path,num_rows ** 2)
        vgg_img_eval, sharp_img_eval, blur_path, sharp_path = loadz_test.loadz_main(file_get=file_get_test, evaluate_img=False)
        # vgg_img_eval, sharp_img_eval, blur_path, sharp_path = loadz_test.loadz_main(file_get='dir_2')

        img_len = len(vgg_img_eval)
        iter = floor(img_len / val_batch_size)
        count = 0
        for i in range(iter):
            vgg_img_eval_iter = vgg_img_eval[count: count+val_batch_size]
            blur_path_eval = blur_path[count: count+val_batch_size]
            vgg_img_eval_iter = vgg_img_eval_iter.cuda(self.rank)

            vgg_feature = help_func.image_noise(vgg_img_eval_iter, device=self.rank, model=self.model_vgg)

            generated_images = self.GAN.G(vgg_img_eval_iter, vgg_feature,  device=self.rank)
            # print(generated_images)
            generated_images = generated_images.detach().cpu()
            # print(len(generated_images))
            generated_images = (generated_images/2 + 0.5)
            
            for j in range(len(generated_images)):
                num = blur_path_eval[j].split('/')[-1]
                print(num)
                torchvision.utils.save_image(generated_images[j], os.path.join(result_img_path, num))
            count += val_batch_size

        print('-----------------')

    @torch.no_grad()
    def val_8000(self, dataloader, num = 0, file_get_test='file', result_img_path=None, val_batch_size=1, trunc = 1.0, valiation=False):
        self.GAN.eval()
        num_rows = self.num_image_tiles
        time_sum = 0
        img_len = 1000
        PSNR_sum = 0
        SSIM_sum = 0

        for ii, item in enumerate(dataloader):
            img_tensor, img_path = item
            vgg_img_eval_iter = img_tensor.cuda(self.rank)

            vgg_feature = help_func.image_noise(vgg_img_eval_iter, device=self.rank, model=self.model_vgg)

            time_start = time.time()
            generated_images = self.GAN.G(vgg_img_eval_iter, vgg_feature,  device=self.rank)
            time_end = time.time()
            print(time_end - time_start)
            time_sum += time_end - time_start

            generated_images = generated_images.detach().cpu()
            # print(len(generated_images))
            generated_images = (generated_images/2 + 0.5)
            
            for j in range(len(generated_images)):
                num = img_path[j].split('/')[-1]
                print(num)
                torchvision.utils.save_image(generated_images[j], os.path.join(result_img_path, num))

        print('单张图片测试耗费时间：{}'.format(time_sum / img_len))
        total = sum([param.nelement() for param in self.GAN.G.parameters()])

        # print(self.GAN.G)
        print("Number of parameter: %.2fM" % (total/1e6))

        print('-----------------')

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # remove any existing files used for fid calculation and recreate directories

        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader_D)
                for k, image in enumerate(real_batch.unbind(0)):
                    filename = str(k + batch_num * self.batch_size)
                    torchvision.utils.save_image(image, str(real_path / f'{filename}.png'))

        # generate a bunch of fake images in results / name / fid_fake

        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.rank)
            
            Blur_path = os.path.join(self.data, 'blur1')
            Sharp_path = os.path.join(self.data, 'sharp1')
            # vgg_img,_ = loadZ(Blur_path, Sharp_path, self.batch_size)
            loadz_test = loadz(Blur_path, Sharp_path, self.batch_size)
            vgg_img,_,_,_ = loadz_test(file_get='dir')

            vgg_img = vgg_img.cuda(self.rank)
            noise = image_noise(vgg_img, device=self.rank)
            for i in range(len(noise)):
                noise[i].cuda(self.rank)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noise, trunc_psi = self.trunc_psi)

            for j, image in enumerate(generated_images.unbind(0)):
                torchvision.utils.save_image(image, str(fake_path / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, noise.device, 2048)

    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi = 0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi = 0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi = trunc_psi)            
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        w = map(lambda t: (S(t[0]), t[1]), style)
        w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi)
        w_styles = styles_def_to_tensor(w_truncated)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi, device=self.rank)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self, i):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('CL', self.content_loss),   
            ('LL', self.Labal_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid)
        ]
            
        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def writer_runs(self, name, value, steps):
        if not exists(self.writer):
            return
        self.writer.add_scalar(name, value, steps)

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.track(value, name = name)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        # rmtree(str(self.runs_dir), True)
        
        self.init_folders()

    def save(self, num, epoch_save=False):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'G_opt.optimizer_state_dict': self.GAN.G_opt.state_dict(),
            'D_opt.optimizer_state_dict': self.GAN.D_opt.state_dict(),
            'version': __version__,
        }

        if self.GAN.fp16:
            save_data['amp'] = amp.state_dict()
 
        self.write_config()
        
        if epoch_save:
            save_path_test = str(self.models_dir / self.name / f'epoch_{num}.pt')
            torch.save(save_data, save_path_test)
            return
            
        torch.save(save_data, self.model_name(num))

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every
        # load_data = torch.load(self.model_name(name))
        load_data = torch.load(self.model_name(name), map_location=torch.device('cpu'))
        
        # del load_data['G_opt.optimizer_state_dict']['state']

        # save_data = {
        #     'GAN': load_data['GAN'],
        #     'G_opt.optimizer_state_dict': load_data['G_opt.optimizer_state_dict']
        # }
        # torch.save(save_data, self.model_name(num))

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])