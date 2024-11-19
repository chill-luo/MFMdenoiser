import random
import openpyxl

import torch


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


# if __name__ == '__main__':
#     stack = torch.tensor([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],
#                           [[-1,-2,-3,-4],[-5,-6,-7,-8],[-9,-10,-11,-12],[-13,-14,-15,-16]],
#                           [[101,102,103,104],[105,106,107,108],[109,110,111,112],[113,114,115,116]]]])
#     print(stack.shape)
#     print(slice_fusion(stack))
#     print(slice_fusion(stack).shape)

#     stack = tiff.imread('data/ConvA/raw/conva.tif')[None, ...]
#     stack = torch.from_numpy(stack.astype(np.float32))
#     substack = stack[:, :2, ...]
#     fusedsub = slice_fusion(substack).numpy()[0].astype(np.uint16)
#     outp = np.concatenate((fusedsub, substack.numpy()[0].astype(np.uint16), fusedsub, fusedsub), axis=0)
#     tiff.imwrite('tmp.tif', outp)