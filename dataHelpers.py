# import netCDF4
import numpy as np
import numpy.ma as ma
import torch
from torch.utils.data import Dataset
import netCDF4 as nc


class ENSODatasetCDF4(Dataset):
    def __init__(self, data_path, target_path=None, is_training=True):
        self.data = nc.Dataset(data_path)
        self.is_training = is_training
        self.target = nc.Dataset(target_path) if self.is_training else None

    def __getitem__(self, index):
        data_np = ma.expand_dims(self.data.variables['sst'][index], -1)
        for key in ['t300', 'ua', 'va']:
            variable = ma.expand_dims(self.data.variables[key][index], -1)
            data_np = ma.concatenate((data_np, variable), axis=-1)

        data_np = data_np.data.transpose(3, 0, 1, 2)

        if self.is_training:
            target_np = self.target.variables['nino'][:].data[index]
            return data_np, data_np[:, 12:, :, :], target_np[12:]
        else:
            return data_np

    def __len__(self):
        return self.data.variables['year'][:].shape[0]


class ENSODataset(Dataset):
    def __init__(self, data_path, target_path=None, is_training=True):
        self.data = np.load(data_path).transpose(0, 4, 1, 2, 3)
        self.is_training = is_training
        self.target = np.load(target_path) if self.is_training else None

    def __getitem__(self, index):
        data_np = self.data[index]

        if self.is_training:
            target_np = self.target[index]
            return data_np, data_np[:, 12:, :, :], target_np[12:]
        else:
            return data_np

    def __len__(self):
        return self.data.shape[0]