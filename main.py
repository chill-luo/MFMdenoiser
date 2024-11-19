import os
import ipdb
import time
import yaml
import argparse
import datetime
import numpy as np
import tifffile as  tiff
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import UNet_n2n_un 
from dataset import ContinuousStackDataset
from dataset import inv_normalize, format
from utils import QueueList, slice_fusion


def train(config):
    print('\n========== Training Stage ==========')
    dateset = ContinuousStackDataset(config)
    trainloader = DataLoader(dateset, batch_size=config['train']['batch_size'], shuffle=True)
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
    prev_time = time.time()
    time_queue = QueueList(max_size=25)
    for epoch in range(config['train']['num_epochs']):
        with tqdm(total=len(trainloader), desc=f"Epoch {epoch + 1}/{config['train']['num_epochs']}", unit='batch') as pbar:
            for i, batch in enumerate(trainloader):
                inds, noisy_torch = batch
                noisy_fused_torch = slice_fusion(noisy_torch, rand=config['train']['random_fus'])
                noisy_fused_torch = noisy_fused_torch.cuda()

                with torch.no_grad():
                    mask = torch.rand_like(noisy_fused_torch)  # uniformly distributed between 0 and 1
                    mask = (mask <  (1 - mask_ratio)).float().cuda()

                output = model(mask * noisy_fused_torch)
                loss = l1_criteron((1 - mask) * output, (1 - mask) * noisy_fused_torch) + \
                       l2_criteron((1 - mask) * output, (1 - mask) * noisy_fused_torch) * 0.5
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
        
        if ((epoch + 1) % config['train']['iter_save'] == 0):
            torch.save(model.state_dict(), os.path.join(config['general']['pth_dir'], config['general']['exp_name'], f'epoch_{epoch+1}.pth'))


def test(config):
    print('\n========== Test Stage ==========')
    dateset = ContinuousStackDataset(config, mode='test')
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

    print('\n===> Show evaluation results:')
    for i, (inds, noisy_original_torch, clean_torch) in enumerate(testloader):
        assert noisy_original_torch.shape[1] == 3

        avg =0.
        for _ in range(config['eval']['num_predictions']):
            with torch.no_grad():
                noisy_fused_1 = slice_fusion(noisy_original_torch[:, :2, :, :], rand=True).cuda()
                noisy_fused_2 = slice_fusion(noisy_original_torch[:, 1:, :, :], rand=True).cuda()
                mask = torch.rand_like(noisy_fused_1)  # uniformly distributed between 0 and 1
                mask = (mask < (1 - mask_ratio)).float().cuda()
                output = (model(mask * noisy_fused_1) + model(mask * noisy_fused_2)) / 2
                to_img = output[0, 0, :, :].detach().cpu().numpy()
            avg += to_img
        denoised_original = avg / float(config['eval']['num_predictions'])
        
        param = dateset.params[inds[0].item()]
        noisy = inv_normalize(noisy_original_torch[0, 1, :, :].cpu().numpy(), param, config['general']['norm'])
        denoised = inv_normalize(denoised_original, param, config['general']['norm'])

        outp = format(np.stack([noisy, denoised], axis=0), dtype=dtype)
        tiff.imwrite(os.path.join(config['general']['save_dir'], config['general']['exp_name'], pth_name, f'{str(i+1).zfill(3)}.tif'), outp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general parameter
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-m", "--mode", type=str, default='train')
    parser.add_argument("-g", "--gpu", type=str, default=0)

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
        train(config)

    if pth_name != 'null':
        config['eval']['load_pth'] = pth_name
    test(config)
