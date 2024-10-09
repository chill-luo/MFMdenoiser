import os
import sys
import time
import ipdb
import yaml
import random
import argparse
import datetime
import numpy as np
import tifffile as  tiff
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import UNet_n2n_un 
from dataset import SingleStackDataset, MultipleStackDataset, ContinuousStackDataset
from dataset import inv_normalize, format
from utils import QueueList, XlsBook, shuffle_image, compare, slice_fusion


def train(config):
    print('\n========== Training Stage ==========')
    dateset = eval(config['general']['dateset_class'])(config)
    trainloader = DataLoader(dateset, batch_size=config['train']['batch_size'], shuffle=True)
    dtype = dateset.dtype
    mask_ratio = config['general']['mask_ratio']
    os.makedirs(os.path.join(config['general']['pth_dir'], config['general']['exp_name']), exist_ok=True)
    with open(os.path.join(config['general']['pth_dir'], config['general']['exp_name'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    print('\n===> Experiment Information')
    print('[General]')
    print(yaml.dump(config['general'], sort_keys=False, default_flow_style=False))
    print('[Train]')
    print(yaml.dump(config['train'], sort_keys=False, default_flow_style=False))

    model = UNet_n2n_un(1, 1).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['train']['num_epochs'])
    l1_criteron = nn.L1Loss(reduction='mean')
    l2_criteron = nn.MSELoss(reduction='mean')

    print('\n===> Start training...')
    shuffle_flag = 'off'
    prev_time = time.time()
    time_queue = QueueList(max_size=25)
    for epoch in range(config['train']['num_epochs']):
        with tqdm(total=len(trainloader), desc=f"Epoch {epoch + 1}/{config['train']['num_epochs']} ({shuffle_flag})", unit='batch') as pbar:
            for i, batch in enumerate(trainloader):
                if (not config['train']['apply_shuffling']) or (epoch <= config['train']['shuffling_epoch'] - 1):
                    inds, noisy_original_torch = batch
                    noisy_original_torch = noisy_original_torch.cuda()
                    noisy_shuffled_torch = noisy_original_torch.detach().clone()
                else:
                    inds, noisy_original_torch, shuffle_mask_torch = batch
                    noisy_original_torch = noisy_original_torch.cuda()
                    shuffle_mask_torch = shuffle_mask_torch.cuda()
                    noisy_orig_np = noisy_original_torch.cpu().squeeze(1).numpy()
                    shuffle_mask_np = shuffle_mask_torch.cpu().squeeze(1).numpy()
                    noisy_shuffled_torch = shuffle_image(noisy_orig_np, shuffle_mask_np, config)
                    noisy_shuffled_torch = noisy_shuffled_torch.cuda()
                
                with torch.no_grad():
                    mask = torch.rand_like(noisy_original_torch)  # uniformly distributed between 0 and 1
                    mask = (mask <  (1. - mask_ratio)).float().cuda()

                output = model(mask * noisy_original_torch)
                loss = l1_criteron((1 - mask) * output, (1 - mask) * noisy_shuffled_torch) + l2_criteron(mask * output, mask * noisy_shuffled_torch) * 0.5
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                iter_time = time.time() - prev_time
                time_queue.add(iter_time)
                iter_time_avg = np.array(time_queue.list).mean()
                time_left = datetime.timedelta(seconds=int(((config['train']['num_epochs'] - epoch) * len(trainloader) - i - 1) * iter_time_avg))
                prev_time = time.time()

                pbar.set_postfix(loss=loss.item(), ETA=time_left)
                pbar.update(1)
        
        if ((epoch + 1) % config['train']['iter_save'] == 0) and ((epoch > config['train']['shuffling_epoch'] - 1) or (not config['train']['apply_shuffling'])):
            torch.save(model.state_dict(), os.path.join(config['general']['pth_dir'], config['general']['exp_name'], f'epoch_{epoch+1}.pth'))
        
        if config['train']['apply_shuffling'] and (epoch == config['train']['shuffling_epoch'] - 1):
            torch.save(model.state_dict(), os.path.join(config['general']['pth_dir'], config['general']['exp_name'], f'epoch_bound.pth'))

            dateset.generate_shuffle_masks(config)
            trainloader = DataLoader(dateset, batch_size=config['train']['batch_size'], shuffle=True)
            shuffle_flag = 'on'
            print('\n===> Continue training with shuffled images...')


def train_custom(config):
    print('\n========== Training Stage ==========')
    dateset = eval(config['general']['dateset_class'])(config)
    trainloader = DataLoader(dateset, batch_size=config['train']['batch_size'], shuffle=True)
    dtype = dateset.dtype
    mask_ratio = config['general']['mask_ratio']
    os.makedirs(os.path.join(config['general']['pth_dir'], config['general']['exp_name']), exist_ok=True)
    with open(os.path.join(config['general']['pth_dir'], config['general']['exp_name'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    print('\n===> Experiment Information')
    print('[General]')
    print(yaml.dump(config['general'], sort_keys=False, default_flow_style=False))
    print('[Train]')
    print(yaml.dump(config['train'], sort_keys=False, default_flow_style=False))

    model = UNet_n2n_un(1, 1).cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['train']['num_epochs'])
    l1_criteron = nn.L1Loss(reduction='mean')
    l2_criteron = nn.MSELoss(reduction='mean')

    print('\n===> Start training...')
    shuffle_flag = 'off'
    prev_time = time.time()
    time_queue = QueueList(max_size=25)
    for epoch in range(config['train']['num_epochs']):
        with tqdm(total=len(trainloader), desc=f"Epoch {epoch + 1}/{config['train']['num_epochs']} ({shuffle_flag})", unit='batch') as pbar:
            for i, batch in enumerate(trainloader):
                inds, noisy_torch = batch
                noisy_fused_torch = slice_fusion(noisy_torch, rand=config['train']['random_fus'])
                noisy_fused_torch = noisy_fused_torch.cuda()

                with torch.no_grad():
                    mask = torch.rand_like(noisy_fused_torch)  # uniformly distributed between 0 and 1
                    mask = (mask <  (1. - mask_ratio)).float().cuda()

                output = model(mask * noisy_fused_torch)
                loss = l1_criteron((1 - mask) * output, (1 - mask) * noisy_fused_torch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                iter_time = time.time() - prev_time
                time_queue.add(iter_time)
                iter_time_avg = np.array(time_queue.list).mean()
                time_left = datetime.timedelta(seconds=int(((config['train']['num_epochs'] - epoch) * len(trainloader) - i - 1) * iter_time_avg))
                prev_time = time.time()

                pbar.set_postfix(loss=loss.item(), ETA=time_left)
                pbar.update(1)
        
        if ((epoch + 1) % config['train']['iter_save'] == 0) and ((epoch > config['train']['shuffling_epoch'] - 1) or not config['train']['apply_shuffling']):
            torch.save(model.state_dict(), os.path.join(config['general']['pth_dir'], config['general']['exp_name'], f'epoch_{epoch+1}.pth'))


def test(config):
    print('\n========== Test Stage ==========')
    dateset = eval(config['general']['dateset_class'])(config, mode='test')
    testloader = DataLoader(dateset, batch_size=1, shuffle=False)
    dtype = dateset.dtype
    mask_ratio = config['general']['mask_ratio']
    pth_name = os.path.splitext(config['eval']['load_pth'])[0]
    os.makedirs(os.path.join(config['general']['save_dir'], config['general']['exp_name'], pth_name), exist_ok=True)

    print('\n===> Experiment Information')
    print('[General]')
    print(yaml.dump(config['general'], sort_keys=False, default_flow_style=False))
    print('[Eval]')
    print(yaml.dump(config['eval'], sort_keys=False, default_flow_style=False))

    model = UNet_n2n_un(1, 1).cuda()
    model.load_state_dict(torch.load(os.path.join(config['general']['pth_dir'], config['general']['exp_name'], config['eval']['load_pth'])))
    model.eval()

    logger = XlsBook(['Number', 'PSNR', 'SSIM'])
    ssim_list = []
    psnr_list = []
    print('\n===> Show evaluation results:')
    for i, (inds, noisy_original_torch, clean_torch) in enumerate(testloader):
        noisy_original_torch = noisy_original_torch.cuda()

        avg =0.
        for _ in range(config['eval']['num_predictions']):
            with torch.no_grad():
                mask = torch.rand_like(noisy_original_torch)  # uniformly distributed between 0 and 1
                mask = (mask < (1. - mask_ratio)).float().cuda()
                output = model(mask * noisy_original_torch)
                to_img = output.detach().cpu().squeeze().numpy()
            avg += to_img
        
        param = dateset.params[inds[0].item()]
        noisy = inv_normalize(noisy_original_torch.cpu().squeeze().squeeze().numpy(), param, config['general']['norm'])
        denoised = inv_normalize(avg / float(config['eval']['num_predictions']), param, config['general']['norm'])

        outp = format(np.stack([noisy, denoised], axis=0), dtype=dtype)
        tiff.imwrite(os.path.join(config['general']['save_dir'], config['general']['exp_name'], pth_name, f'{str(i+1).zfill(3)}.tif'), outp)

        if torch.sum(clean_torch).item() > 0:
            clean = clean_torch.squeeze().squeeze().numpy().astype(dtype)
            ssim, psnr = compare(denoised, clean, alignment=config['eval']['alignment'])
            print(f'image {i+1}: PSNR={psnr:.2f} & SSIM={ssim:.4f}')
            logger.write([i+1, psnr, ssim])
            ssim_list.append(ssim)
            psnr_list.append(psnr)
    
    if torch.sum(clean_torch).item() > 0:
        print(f'\nAverage: PSNR={np.mean(psnr_list):.2f} & SSIM={np.mean(ssim_list):.4f}')
        logger.write(['AVG', np.mean(psnr_list), np.mean(ssim_list)])
        logger.save(os.path.join(config['general']['save_dir'], config['general']['exp_name'], pth_name, '_results.xlsx'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general parameter
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-m", "--mode", type=str, default='train')
    parser.add_argument("-g", "--gpu", type=str, default=0)

    # train parameter
    parser.add_argument("-t", "--custum_train", action='store_true')

    # test parameter
    parser.add_argument("-n", "--exp_name", type=str, default='null')
    parser.add_argument("-p", "--pth", type=str, default='null')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
    mode = args.mode
    pth_name = args.pth

    if mode == 'train' or args.exp_name == 'null':
        config_file = args.config
    else:
        config_file = os.path.join('pths', args.exp_name, 'config.yaml')

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if mode == 'train':
        if args.custum_train:
            train_custom(config)
        else:
            train(config)

    if pth_name != 'null':
        config['eval']['load_pth'] = pth_name
    test(config)
