import netCDF4
import numpy as np
import numpy.ma as ma
import torch
from torch.utils.data import Dataset

class ENSODataset(Dataset):

    def __init__(self, data_path, target_path, is_training=True):
        self.data = netCDF4.Dataset(data_path)
        self.is_training = is_training
        self.target = netCDF4.Dataset(target_path) if self.is_training else None
        

    def __getitem__(self, index):
        data_np = ma.expand_dims(self.data.variables['sst'][index], -1)
        for key in ['t300', 'ua', 'va']:
            variable = ma.expand_dims(self.data.variables[key][index], -1)
            data_np = ma.concatenate((data_np,variable), axis=-1)

        data_np = data_np.data.transpose(3,0,1,2)
        
        if self.is_training:
            target_np = self.target.variables['nino'][:].data[index]
            return data_np, data_np[:,12:,:,:],target_np[12:]
        else:
            return data_np

    def __len__(self):
        return self.data.variables['year'][:].shape[0]