import os
import ipdb
import yaml
import random
import numpy as np
import tifffile as tiff

import torch
from torch.utils.data import Dataset, DataLoader


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


class ContinuousStackDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.norm = config['general']['norm']
        self.crop_size = config['train']['crop_size']
        self.fus_num = config['train']['fus_num'] if mode == 'train' else 3
        self.mode = mode
        
        print('===> Data loaded.')
        self.params = []
        self.raw_list, self.dtype = self.scan_stack_files(config['general']['raw_dir'], norm=config['general']['norm'], info=True)

        if config['general']['gt_dir'] is not None:
            self.gt_list, _ = self.scan_stack_files(config['general']['gt_dir'], norm=None)
        else:
            self.gt_list = np.zeros(len(self.raw_list), dtype=np.float32)

    def __len__(self):
        return len(self.raw_list)

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
            stk = self.raw_list[idx]
            gt = self.gt_list[idx][1]         
            if (stk.shape[-2]%32 != 0) or (stk.shape[-1]%32 != 0):
                stk = center_crop(stk)
                gt = center_crop(gt)
            stk = torch.from_numpy(stk).float()
            gt = torch.from_numpy(gt).float().unsqueeze(0)
            return idx, stk, gt
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