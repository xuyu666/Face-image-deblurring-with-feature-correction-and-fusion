import os
import random
from tqdm import tqdm
from functools import wraps

from pipeline import Trainer
from help_func import *

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import time
import pytz
from datetime import datetime
from pytz import timezone
from torch.utils import data as dataclass

import numpy as np

import help_class
import help_func

from tensorboardX import SummaryWriter

def cast_list(el):
    return el if isinstance(el, list) else [el]

def timestamped_filename(prefix = 'generated-'):
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(tz)
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def run_training(rank, world_size, model_args, kernel_data, data, iswriter, load_from, new, num_train_steps, name, seed, runs_dir, blur_path_valiation, sharp_path_valiation, result_img_path_ori):
    # is_main = rank == 0
    kernel_data = kernel_data
    is_ddp = world_size > 1
    
    if is_ddp:
        set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(
        is_ddp = is_ddp,
        rank = rank,
        world_size = world_size
    )

    if iswriter==True:
        tz = pytz.timezone('Asia/Shanghai')
        now = datetime.now(tz)
        timestamp = now.strftime("%Y-%m-%d-%H:%M-%S")
        writer = SummaryWriter(os.path.join(runs_dir, timestamp))
    else: 
        writer = None

    model = Trainer(writer, **model_args)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    file_get_input = 'file'
    val_batch_size_ori = 1
    GTPath = sharp_path_valiation

    mk_dir(path=blur_path_valiation)
    mk_dir(path=sharp_path_valiation)
    if os.path.exists(result_img_path_ori):
        shutil.rmtree(path=result_img_path_ori)
    mk_dir(path=result_img_path_ori)

    model.valiation(blur_path_valiation, sharp_path_valiation, 0, file_get_test = file_get_input, result_img_path=result_img_path_ori, val_batch_size=val_batch_size_ori, valiation=True)

    bluringImgPath = result_img_path_ori
    PSNR_avg, SSIM_avg = help_func.metrics(bluringImgPath, GTPath)
    print('PSNR:{}'.format(PSNR_avg))
    print('------------')
    
def train_from_folder(
    blur_path_valiation = 'data/test/blur',
    sharp_path_valiation = 'data/test/gt',
    result_img_path_ori = 'data/test/result',
    data = './data',
    kernel_data = './',
    results_dir = './results',
    models_dir = './models',
    runs_dir='./runs/logs',
    name = 'default',
    iswriter = False,
    new = False,
    load_from = -1,
    image_size = 128,
    network_capacity = 32,
    fmap_max = 512,
    transparent = False,
    batch_size = 5,
    gradient_accumulate_every = 6,
    num_train_steps = 1500000,
    learning_rate = 2e-4,
    lr_mlp = 0.1,
    ttur_mult = 1.5,
    rel_disc_loss = False,
    num_workers =  None,
    save_every = 1000,
    evaluate_every = 1000,
    generate = False,
    num_generate = 1,
    generate_interpolation = False,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = 8,
    trunc_psi = 0.75,
    mixed_prob = 0.9,
    fp16 = False,
    no_pl_reg = False,
    cl_reg = False,
    fq_layers = [],
    fq_dict_size = 256,
    attn_layers = [1, 2],
    no_const = False,
    aug_prob = 0.,
    aug_types = ['translation', 'cutout'],
    top_k_training = False,
    generator_top_k_gamma = 0.99,
    generator_top_k_frac = 0.5,
    dual_contrast_loss = False,
    dataset_aug_prob = 0.,
    multi_gpus = False,
    calculate_fid_every = None,
    calculate_fid_num_images = 12800,
    clear_fid_cache = False,
    seed = 42,
    log = False
):
    model_args = dict(
        data = data,
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        runs_dir = runs_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        network_capacity = network_capacity,
        fmap_max = fmap_max,
        transparent = transparent,
        lr = learning_rate,
        lr_mlp = lr_mlp,
        ttur_mult = ttur_mult,
        rel_disc_loss = rel_disc_loss,
        num_workers = num_workers,
        save_every = save_every,
        evaluate_every = evaluate_every,
        num_image_tiles = num_image_tiles,
        trunc_psi = trunc_psi,
        fp16 = fp16,
        no_pl_reg = no_pl_reg,
        cl_reg = cl_reg,
        fq_layers = fq_layers,
        fq_dict_size = fq_dict_size,
        attn_layers = attn_layers,
        no_const = no_const,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        top_k_training = top_k_training,
        generator_top_k_gamma = generator_top_k_gamma,
        generator_top_k_frac = generator_top_k_frac,
        dual_contrast_loss = dual_contrast_loss,
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache,
        mixed_prob = mixed_prob,
        log = log
    )

    run_training(run_gpu, 1, model_args, kernel_data, data, iswriter, load_from, new, num_train_steps, name, seed, runs_dir, blur_path_valiation, sharp_path_valiation, result_img_path_ori)


if __name__ == '__main__':

    # model_file_name
    py_name = 'val'

    # val path setting
    blur_path_valiation = 'data/test/blur'
    sharp_path_valiation = 'data/test/gt'
    result_img_path_ori = 'data/test/result'

    # gpu setting
    run_gpu = 0

    train_from_folder(blur_path_valiation, sharp_path_valiation, result_img_path_ori, name=py_name)

    

    