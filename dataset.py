import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import numpy as np


class dSpritesDataset(Dataset):
    def __init__(self):
        #f = h5py.File(
        #    'dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5')
        f = np.load('dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        idxs = torch.randperm(737280)[:10000].numpy()
        idxs = np.sort(idxs)
        #print('shit')
        #__imgs = f['imgs'][idxs]
        #print('ok')
        self.imgs = torch.tensor(
            f['imgs'][idxs],
            dtype=torch.float
        ).reshape((-1, 1, 64, 64)).to('cuda:0')
        self.latents = torch.tensor(f['latents_values'][idxs])
        self.latents_classes = torch.tensor(f['latents_classes'][idxs])
        f.close()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]

    def get_similar(self, idx):
        pass
