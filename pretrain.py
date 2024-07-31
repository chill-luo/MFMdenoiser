import numpy as np
import tifffile as  tiff
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import UNet_n2n_un 
from dataset import SingleStackDateset, inv_normalize, format


def pre_train(config):
    print('Starting pre-training...')
    dateset = SingleStackDateset(config)
    trainloader = DataLoader(dateset, batch_size=config['batch_size'], shuffle=True)
    testloader = DataLoader(dateset, batch_size=1, shuffle=False)
    dtype = dateset.dtype

    estimated_std_list = []
    for mask_ratio in [0.8, 0.2]:
        model = UNet_n2n_un(1, 1).cuda()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epochs'])
        l1_criteron = nn.L1Loss(reduction='mean')

        print('\n===> Pre-training with mask ratio: ', mask_ratio)
        for i in range(config['num_epochs']):
            with tqdm(total=len(trainloader), desc=f"Epoch {i + 1}/{config['num_epochs']}", unit='batch') as pbar:
                for noisy_img in trainloader:
                    noisy_img = noisy_img.cuda()

                    with torch.no_grad():
                        mask = torch.rand_like(noisy_img)  # uniformly distributed between 0 and 1
                        mask = (mask < (1. - mask_ratio)).float().cuda()

                    output = model(mask * noisy_img)
                    loss = l1_criteron((1 - mask) * output, (1 - mask) * noisy_img) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)
        
        print('\n===> Estimating noise variance based on pre training results')
        with torch.no_grad():
            model.eval()
            estimated_std = []
            j = 0
            for noisy_img in tqdm(testloader):
                avg = 0.
                noisy_img = noisy_img.cuda()
                for _ in range(config['num_predictions']):
                    mask = torch.rand_like(noisy_img)
                    mask = (mask < (1. - mask_ratio)).float().cuda()

                    output = model(mask * noisy_img)
                    to_img = output.detach().cpu().squeeze().squeeze().numpy()
                    avg += to_img

                noisy_orig_np = inv_normalize(noisy_img.cpu().squeeze().squeeze().numpy(), dateset.param, config['norm'])
                denoised_np = inv_normalize(avg/float(config['num_predictions']), dateset.param, config['norm'])

                outp = format(np.stack([noisy_orig_np, denoised_np], axis=0), dtype=dtype)
                tiff.imwrite(f'tmp/denoised_img_{mask_ratio}_{j}.tiff', outp)
                j += 1

                estimated_std.append(np.std(denoised_np - noisy_orig_np))
            estimated_std_list.append(estimated_std)
        
    return estimated_std_list
