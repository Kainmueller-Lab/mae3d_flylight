import numpy as np
import zarr
from glob import glob
import torch
from torch.utils.data import Dataset


class FlylightDataset(Dataset):

    def __init__(self, root_dir, input_size, transform=None):
        self.root_dir = root_dir
        self.input_size = input_size
        self.transform = transform
        self.samples = glob(self.root_dir + "/**/*.zarr")
        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # load zarr
        zraw =  zarr.open(sample, "r", path="volumes/raw")
        
        # get random crop
        d, h, w = zraw.shape[-3:]
        new_d = new_h = new_w = self.input_size
        d_start = np.random.randint(0, d - new_d + 1)
        h_start = np.random.randint(0, h - new_h + 1)
        w_start = np.random.randint(0, w - new_w + 1)

        raw = np.array(zraw[...,
            d_start:d_start + new_d,
            h_start:h_start + new_h,
            w_start:w_start + new_w])
        
        # normalize
        raw = np.clip(raw, 0, 1500)
        raw = raw / 1500.

        if self.transform:
            raw = self.transform(raw)

        return raw

