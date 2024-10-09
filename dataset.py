import os
import ipdb
import yaml
import random
import numpy as np
import tifffile as tiff
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import UNet_n2n_un 
from utils import calculate_sliding_std, get_shuffling_mask


def normalize(x, mode='submean'):
    dtype = x.dtype
    x = x.astype(np.float32)
    if mode == 'submean':
        m = x.mean()
        x = x - m
        return x, m
    elif mode == 'standard':
        m = x.mean()
        s = x.std()
        x = (x - m) / (s + 1e-6)
        return x, (m, s)
    elif mode == 'hard':
        p1 = x.min()
        p2 = x.max()
        x = (x - p1) / (p2 - p1)
        return x, (p1, p2)
    elif mode == 'soft':
        r = 255 if dtype == np.uint8 else 65535
        x = x / r
        return x, r
    elif mode == 'none':
        return x, 0
    else:
        raise ValueError(f'No such normlize mode: {mode}')
    

def inv_normalize(x, param, mode='submean'):
    if mode == 'submean':
        x = x + param
    elif mode == 'standard':
        x = x * param[1] + param[0]
    elif mode == 'hard':
        x = x * (param[1] - param[0]) + param[0]
    elif mode == 'soft':
        x = x * param
    elif mode == 'none':
        pass
    else:
        raise ValueError(f'No such normlize mode: {mode}')
    return x


def random_crop(x_list, crop_size):
    if not isinstance(x_list, list):
        x_list = [x_list]

    h, w = x_list[0].shape[-2:]
    pin_h = random.randint(0, h-crop_size)
    pin_w = random.randint(0, w-crop_size)

    y_list = []
    for x in x_list:
        y_list.append(x[..., pin_h:pin_h+crop_size, pin_w:pin_w+crop_size])
    return y_list


def center_crop(x, crop_size=None):
    h, w = x.shape[-2:]
    if crop_size is None:
        h_new = 32 * (h // 32)
        w_new = 32 * (w // 32)
    else:
        h_new = crop_size
        w_new = crop_size
    
    y = x[..., h//2-h_new//2:h//2+h_new//2, w//2-w_new//2:w//2+w_new//2]
    return y


def format(x, dtype):
    if dtype == np.uint8:
        depth = 255
    elif dtype == np.uint16:
        depth = 65535
    else:
        raise ValueError("Unsupported image type: {}.".format(dtype))
    
    x = np.clip(x, 0, depth)
    x = x.astype(dtype)
    return x


class SingleStackDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.norm = config['general']['norm']
        self.crop_size = config['train']['crop_size']
        self.mode = mode

        print('===> Data loaded.')
        self.stack = tiff.imread(config['general']['raw_path'])
        print(f'Stack: {self.stack.shape}')
        self.dtype = self.stack.dtype
        self.stack, param = normalize(self.stack, self.norm)
        self.params = [param] * self.stack.shape[0]
        self.shuffle_mask_stack = None

        if config['general']['gt_path'] is not None:
            self.gt_stack = tiff.imread(config['general']['gt_path'])
            self.gt_stack = self.gt_stack.astype(np.float32)
            if self.gt_stack.ndim == self.stack.ndim - 1:
                self.gt_stack = np.expand_dims(self.gt_stack, axis=0).repeat(self.stack.shape[0], axis=0)
            elif self.gt_stack.ndim != self.stack.ndim:
                raise ValueError('The dimension of gt and raw should be the same.')
        else:
            self.gt_stack = np.zeros(self.stack.shape[0], dtype=np.float32)

    def __len__(self):
        return self.stack.shape[0]

    def __getitem__(self, idx):
        if self.mode == 'train':
            if self.shuffle_mask_stack is None:
                img = self.stack[idx]
                if self.crop_size is not None:
                    img = random_crop([img], self.crop_size)[0]
                elif (img.shape[-2]%32 != 0) or (img.shape[-1]%32 != 0):
                    img = center_crop(img)
                img = torch.from_numpy(img).float().unsqueeze(0)
                return idx, img
            else:
                img = self.stack[idx]
                mask = self.shuffle_mask_stack[idx]
                if self.crop_size is not None:
                    img, mask = random_crop([img, mask], self.crop_size)
                elif (img.shape[-2]%32 != 0) or (img.shape[-1]%32 != 0):
                    img = center_crop(img)
                    mask = center_crop(mask)
                img = torch.from_numpy(img).float().unsqueeze(0)
                mask = torch.from_numpy(mask).float().unsqueeze(0)
                return idx, img, mask
        elif self.mode == 'test':
            img = self.stack[idx]
            gt = self.gt_stack[idx]
            if (img.shape[-2]%32 != 0) or (img.shape[-1]%32 != 0):
                img = center_crop(img)
                gt = center_crop(gt)
            img = torch.from_numpy(img).float().unsqueeze(0)
            gt = torch.from_numpy(gt).float().unsqueeze(0)
            return idx, img, gt
        else:
            raise ValueError(f'No such mode: {self.mode}')

    def generate_shuffle_masks(self, config):
        upsampler = nn.Upsample(scale_factor=config['train']['std_kernel_size'], mode='nearest')
        self.shuffle_mask_stack = np.zeros_like(self.stack)
        os.makedirs(os.path.join(config['general']['save_dir'], config['general']['exp_name'], 'masks'), exist_ok=True)

        model = UNet_n2n_un(1, 1).cuda()
        model.load_state_dict(torch.load(os.path.join(config['general']['pth_dir'], config['general']['exp_name'], f'epoch_bound.pth')))
        model.eval()

        print('\n===> Generating shuffling masks...')
        for i in tqdm(range(self.stack.shape[0])):
            noisy_original_torch = torch.from_numpy(self.stack[i]).float().unsqueeze(0).unsqueeze(0).cuda()

            avg =0.
            for _ in range(config['eval']['num_predictions']):
                with torch.no_grad():
                    mask = torch.rand_like(noisy_original_torch)  # uniformly distributed between 0 and 1
                    mask = (mask < (1. - config['general']['mask_ratio'])).float().cuda()
                    output = model(mask * noisy_original_torch)
                    to_img = output.detach().cpu().squeeze().squeeze().numpy()
                avg += to_img
            
            denoised = inv_normalize(avg / float(config['eval']['num_predictions']), self.params[i], config['general']['norm'])
            std_map_torch = calculate_sliding_std(torch.from_numpy(denoised).unsqueeze(0).unsqueeze(0).cuda(), config['train']['std_kernel_size'], 1)
            shuffling_mask = get_shuffling_mask(std_map_torch,  config['train']['masking_threshold'])
            self.shuffle_mask_stack[i] = shuffling_mask

            tiff.imwrite(os.path.join(config['general']['save_dir'], config['general']['exp_name'], 'masks', f'{str(i+1).zfill(3)}_image.tif'), denoised)
            tiff.imwrite(os.path.join(config['general']['save_dir'], config['general']['exp_name'], 'masks', f'{str(i+1).zfill(3)}_mask.tif'), shuffling_mask)


class MultipleStackDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.norm = config['general']['norm']
        self.crop_size = config['train']['crop_size']
        self.mode = mode
        
        print('===> Data loaded.')
        self.params = []
        self.raw_list, self.dtype = self.scan_stack_files(config['general']['raw_dir'], norm=config['general']['norm'], info=True)
        self.shuffle_mask_list = None

        if config['general']['gt_dir'] is not None:
            self.gt_list, _ = self.scan_stack_files(config['general']['gt_dir'], norm=None)
        else:
            self.gt_list = np.zeros(len(self.raw_list), dtype=np.float32)

    def __len__(self):
        return len(self.raw_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            if self.shuffle_mask_list is None:
                img = self.raw_list[idx]
                if self.crop_size is not None:
                    img = random_crop([img], self.crop_size)[0]
                elif (img.shape[-2]%32 != 0) or (img.shape[-1]%32 != 0):
                    img = center_crop(img)
                img = torch.from_numpy(img).float().unsqueeze(0)
                return idx, img           
            else:
                img = self.raw_list[idx]
                mask = self.shuffle_mask_list[idx]
                if self.crop_size is not None:
                    img, mask = random_crop([img, mask], self.crop_size)
                elif (img.shape[-2]%32 != 0) or (img.shape[-1]%32 != 0):
                    img = center_crop(img)
                    mask = center_crop(mask)
                img = torch.from_numpy(img).float().unsqueeze(0)
                mask = torch.from_numpy(mask).float().unsqueeze(0)
                return idx, img, mask           
        elif self.mode == 'test':
            img = self.raw_list[idx]
            gt = self.gt_list[idx]
            if (img.shape[-2]%32 != 0) or (img.shape[-1]%32 != 0):
                img = center_crop(img)
                gt = center_crop(gt)
            img = torch.from_numpy(img).float().unsqueeze(0)
            gt = torch.from_numpy(gt).float().unsqueeze(0)
            return idx, img, gt       
        else:
            raise ValueError(f'No such mode: {self.mode}')
    
    def scan_stack_files(self, dir, norm=None, info=False):
        files = os.listdir(dir)
        files.sort()
        stack_list = []
        for i, f in enumerate(files):
            if '.tif' not in f:
                continue
            stack = tiff.imread(os.path.join(dir, f))
            dtype = stack.dtype
            if info:
                print(f'Stack {i+1}: {stack.shape}')

            if norm is not None:
                stack, param = normalize(stack, norm)
                self.params += [param] * stack.shape[0]

            for j in range(stack.shape[0]):
                stack_list.append(stack[j].astype(np.float32))
            
        return stack_list, dtype

    def generate_shuffle_masks(self, config):
        upsampler = nn.Upsample(scale_factor=config['train']['std_kernel_size'], mode='nearest')
        self.shuffle_mask_list = []
        os.makedirs(os.path.join(config['general']['save_dir'], config['general']['exp_name'], 'masks'), exist_ok=True)

        model = UNet_n2n_un(1, 1).cuda()
        model.load_state_dict(torch.load(os.path.join(config['general']['pth_dir'], config['general']['exp_name'], f'epoch_bound.pth')))
        model.eval()

        print('\n===> Generating shuffling masks...')
        for i in tqdm(range(len(self.raw_list))):
            noisy_original_torch = torch.from_numpy(self.raw_list[i]).float().unsqueeze(0).unsqueeze(0).cuda()

            avg =0.
            for _ in range(config['eval']['num_predictions']):
                with torch.no_grad():
                    mask = torch.rand_like(noisy_original_torch)  # uniformly distributed between 0 and 1
                    mask = (mask < (1. - config['general']['mask_ratio'])).float().cuda()
                    output = model(mask * noisy_original_torch)
                    to_img = output.detach().cpu().squeeze().squeeze().numpy()
                avg += to_img
            
            denoised = inv_normalize(avg / float(config['eval']['num_predictions']), self.params[i], config['general']['norm'])
            std_map_torch = calculate_sliding_std(torch.from_numpy(denoised).unsqueeze(0).unsqueeze(0).cuda(), config['train']['std_kernel_size'], 1)
            shuffling_mask = get_shuffling_mask(std_map_torch,  config['train']['masking_threshold'])
            self.shuffle_mask_list.append(shuffling_mask)

            tiff.imwrite(os.path.join(config['general']['save_dir'], config['general']['exp_name'], 'masks', f'{str(i+1).zfill(3)}_image.tif'), format(denoised, self.dtype))
            tiff.imwrite(os.path.join(config['general']['save_dir'], config['general']['exp_name'], 'masks', f'{str(i+1).zfill(3)}_mask.tif'), shuffling_mask)


class ContinuousStackDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.norm = config['general']['norm']
        self.crop_size = config['train']['crop_size']
        self.fus_num = config['train']['fus_num']
        self.mode = mode
        
        print('===> Data loaded.')
        self.params = []
        self.raw_list, self.dtype = self.scan_stack_files(config['general']['raw_dir'], norm=config['general']['norm'], info=True)

        if config['general']['gt_dir'] is not None:
            self.gt_list, _ = self.scan_stack_files(config['general']['gt_dir'], norm=None)
        else:
            self.gt_list = np.zeros(len(self.raw_list), dtype=np.float32)

    def __len__(self):
        if self.mode == 'train':
            return len(self.raw_list)
        else:
            return len(self.raw_list) + self.fus_num - 1

    def __getitem__(self, idx):
        if self.mode == 'train':
            stk = self.raw_list[idx]
            if self.crop_size is not None:
                stk = random_crop(stk, self.crop_size)[0]
            elif (stk.shape[-2]%32 != 0) or (stk.shape[-1]%32 != 0):
                stk = center_crop(stk)
            stk = torch.from_numpy(stk).float()
            return idx, stk
        elif self.mode == 'test':
            if idx < len(self.raw_list):
                img = self.raw_list[idx][0]
                gt = self.gt_list[idx][0]
            else:
                img = self.raw_list[len(self.raw_list)-1][idx-len(self.raw_list)+1]
                gt = self.gt_list[len(self.raw_list)-1][idx-len(self.raw_list)+1]            
            if (img.shape[-2]%32 != 0) or (img.shape[-1]%32 != 0):
                img = center_crop(img)
                gt = center_crop(gt)
            img = torch.from_numpy(img).float().unsqueeze(0)
            gt = torch.from_numpy(gt).float().unsqueeze(0)
            return idx//self.fus_num, img, gt
        else:
            raise ValueError(f'No such mode: {self.mode}')
    
    def scan_stack_files(self, dir, norm=None, info=False):
        files = os.listdir(dir)
        files.sort()
        stack_list = []
        for i, f in enumerate(files):
            if '.tif' not in f:
                continue
            stack = tiff.imread(os.path.join(dir, f))
            dtype = stack.dtype
            if info:
                print(f'Stack {i+1}: {stack.shape}')

            if norm is not None:
                stack, param = normalize(stack, norm)
                self.params += [param] * (stack.shape[0] - self.fus_num+1)

            for j in range(stack.shape[0]-self.fus_num+1):
                stack_list.append(stack[j:j+self.fus_num].astype(np.float32))
            
        return stack_list, dtype


if __name__ == '__main__':
    with open('configs/fmdwf.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['train']['mix_num'] = 3
    
    dataset = ContinuousStackDataset(config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    for batch in dataloader:
        idx, image = batch
        ipdb.set_trace()
        # print(idx)
        # print(len(dataset.params))
        # print(dataset.params)
        # print(image.shape)