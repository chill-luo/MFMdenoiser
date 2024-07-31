import ipdb
import random
import openpyxl
import numpy as np
import tifffile as tiff
from einops import rearrange
from scipy.optimize import leastsq
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch


def get_conv(kernel_size=3, stride=1):
    padding_size = (kernel_size - 1) // 2
    conv = torch.nn.Conv2d(in_channels=1, 
                           out_channels=1, 
                           kernel_size=kernel_size, 
                           stride=stride, 
                           padding=padding_size, 
                           bias=False,
                           padding_mode='reflect')
    conv.weight = torch.nn.Parameter((torch.ones((1, 1, kernel_size, kernel_size))/(1.0 * kernel_size ** 2)).cuda())
    return conv


def smooth(noisy, kernel_size=3, stride=1):
    conv = get_conv(kernel_size, stride)
    b, c, h, w = noisy.shape
    smoothed = conv(noisy.view(-1, 1, h, w))
    _, _, new_h, new_w = smoothed.shape     
    return smoothed.view(1, c, new_h, new_w).detach()


def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_sliding_std(img, kernel_size=3, stride=1):   
    slided_mean = smooth(img, kernel_size, stride)
    # mean_upsampled = upsampler(slided_mean)
    variance = smooth((img - slided_mean)**2, kernel_size, stride)
    # upsampled_variance = upsampler(variance)
    return variance.sqrt()


def shuffle_input(img, indices, mask, c, h, w, k):
    if c == 1:
        img_torch = torch.from_numpy(img).unsqueeze(0)
    else:
        img_torch = torch.from_numpy(img)
    mask_torch = torch.from_numpy(mask).unsqueeze(0).repeat(c, 1, 1)
    img_torch_rearranged = rearrange(img_torch.unsqueeze(1), 'c 1 (h1 h) (w1 w) -> c (h1 w1) h w ', h1=h//k, w1=w//k) # (c H//k W//k k k)
    mask_torch_rearranged = rearrange(mask_torch.unsqueeze(1), 'c 1 (h1 h) (w1 w) -> c (h1 w1) h w ', h1=h//k, w1=w//k)
    img_torch_rearranged = img_torch_rearranged.view(c, -1, k*k)# (c H//k*W//k k*k)
    mask_torch_rearranged, _ = torch.max(mask_torch_rearranged.view(c, -1, k*k), 2, keepdim=True)
    img_torch_reordered = torch.gather(img_torch_rearranged.clone(), dim=-1, index=indices).clone()
    img_torch_reordered_v2 = img_torch_reordered.view(c, -1, k*k)
    # Shuffle the image only at the flat regions (where mask = 0)
    img_torch_final = mask_torch_rearranged * img_torch_rearranged + (1 - mask_torch_rearranged) * img_torch_reordered_v2
    img_torch_final = img_torch_final.view(c, -1, k, k)    
    img_torch_final_v2 = rearrange(img_torch_final, 'c (h1 w1) h w -> c 1 (h1 h) (w1 w) ', h1=h//k, w1=w//k)
    return img_torch_final_v2.squeeze().cpu().numpy() 


def get_shuffling_mask(std_map_torch, threshold=0.5):
    std_map = std_map_torch.cpu().numpy().squeeze()
    normalized = std_map/std_map.max()
    thresholded = np.zeros_like(normalized)
    thresholded[normalized >= threshold] = 1.
    return thresholded


def generate_random_permutation(h, w, c, factor):
    d1, d2, d3 = c, (h//factor)*(w//factor), factor*factor
    permutaion_indices = torch.argsort(torch.rand(1, d2, d3), dim=-1)
    permutaion_indices = permutaion_indices.repeat(d1, 1, 1)
    reverse_permutation_indices = torch.argsort(permutaion_indices, dim=-1)

    return permutaion_indices, reverse_permutation_indices


def get_mask_ratio(estimated_std_list, config):
    diff = np.mean(np.abs(np.array(estimated_std_list[0])- np.array(estimated_std_list[1])))

    if diff > config['epsilon_high']:
        apply_local_shuffling=True
        mask_ratio = config['mask_high']
    elif diff < config['epsilon_low']:
        apply_local_shuffling = False
        mask_ratio = config['mask_low']
    else:
        apply_local_shuffling = False
        mask_ratio = config['mask_medium']
    return mask_ratio, apply_local_shuffling


def shuffle_image(images, masks, config):
    n, h, w = images.shape
    noisy_shuffled_torch = torch.zeros(n, 1, h, w)

    i = 0
    for image, mask in zip(images, masks):
        permutation_indices, _ = generate_random_permutation(h, w, 1, config['train']['shuffling_tile_size'])
        noisy_shuffled_np= shuffle_input(image,  permutation_indices, mask=mask, c=1, h=h, w=w, k=config['train']['shuffling_tile_size'])
        noisy_shuffled_torch[i] = torch.from_numpy(noisy_shuffled_np).unsqueeze(0)
        i += 1

    return noisy_shuffled_torch


def slice_fusion(stack, rand=False):
    n, d, h, w = stack.shape
    fused_stack = torch.zeros(n, 1, h, w)
    fuse_mask_base = torch.zeros(h+2*d-2, w)
    fuse_mask = torch.zeros(d, h, w)

    for i in range(h+2*d-2):
        s = i % d
        fuse_mask_base[i, s::d] = 1
    
    s = random.randint(0, d-1) if rand else 0
    
    for j in range(d):
        fuse_mask[j] = fuse_mask_base[s+j:s+j+h, :]
    
    for k in range(n):
        fused_stack[k] = torch.sum(stack[k] * fuse_mask, dim=0, keepdim=True)

    return fused_stack


def simpleNormalize(x):
    tmp = x.astype(np.float32).flatten()
    tmp.sort()
    p1 = tmp[int(len(tmp) * 0.001)]
    p2 = tmp[int(len(tmp) * 0.999)]
    x = (x - p1) / (p2 - p1 + 1e-8)       
    return x


def consistent_translate(x, y, norm):

    def func(p, x):
        k, b = p
        return k * x + b
    
    def error(p, x, y):
        return func(p, x) - y
    
    if norm:
        x = simpleNormalize(x)
        y = simpleNormalize(y)
    else:
        x = x.astype(np.float32)
        y = y.astype(np.float32)
    
    factor = y.mean() / (x.mean() + 1e-8)
    k_init = np.random.normal(factor, 1)
    b_init = np.random.normal(0, 1)
    k, b = leastsq(error, [k_init, b_init], args=(x.flatten(), y.flatten()))[0]
    
    x_t = k * x + b
    return x_t, y, (k, b)


def compare(image, target, alignment=False):
    def vrange(x):
        if x.max() <= 255:
            return 255
        else:
            return x.max() - x.min()

    if alignment:
        image_ssim, target_ssim, _ = consistent_translate(image, target, norm=True)
        image_psnr, target_psnr, _ = consistent_translate(image, target, norm=False)
        score_ssim = ssim(image_ssim, target_ssim, data_range=1)
        score_psnr = psnr(target_psnr, image_psnr, data_range=vrange(target_psnr))
    else:
        score_ssim = ssim(image.astype(np.float32), target.astype(np.float32), data_range=vrange(target))
        score_psnr = psnr(target.astype(np.float32), image.astype(np.float32), data_range=vrange(target))

    return score_ssim, score_psnr


class QueueList():
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.list = []

    def add(self, x):
        if len(self.list) == self.max_size:
            self.list.pop(0)
        self.list.append(x)

    def remove(self, i):
        self.list.pop(i)


class XlsBook():
    def __init__(self, labels, sheet_name='log'):
        self.labels = labels
        self.book = openpyxl.Workbook()
        self.sheet = self.book.create_sheet(sheet_name, 0)
        self.sheet.append(labels)

    def write(self, values):
        if len(values) != len(self.labels):
            raise ValueError('Inputs of logger does not match the length of the labels.')
        self.sheet.append(values)

    def save(self, save_path):
        self.book.save(save_path)


if __name__ == '__main__':
    # stack = torch.tensor([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],
    #                       [[-1,-2,-3,-4],[-5,-6,-7,-8],[-9,-10,-11,-12],[-13,-14,-15,-16]],
    #                       [[101,102,103,104],[105,106,107,108],[109,110,111,112],[113,114,115,116]]]])
    # print(stack.shape)
    # print(slice_fusion(stack))
    # print(slice_fusion(stack).shape)

    # stack = tiff.imread('data/ConvA/raw/conva.tif')[None, ...]
    # stack = torch.from_numpy(stack.astype(np.float32))
    # substack = stack[:, :2, ...]
    # fusedsub = slice_fusion(substack).numpy()[0].astype(np.uint16)
    # outp = np.concatenate((fusedsub, substack.numpy()[0].astype(np.uint16), fusedsub, fusedsub), axis=0)
    # tiff.imwrite('tmp.tif', outp)

    outp = tiff.imread('results/conva_base_09/epoch_200/001.tif')
    denoised = outp[1]
    # denoised = tiff.imread('data/ConvA/raw/conva.tif')[0]
    gt = tiff.imread('data/ConvA/gt_single.tif')
    score_ssim, score_psnr = compare(denoised, gt)
    print(score_ssim, score_psnr)