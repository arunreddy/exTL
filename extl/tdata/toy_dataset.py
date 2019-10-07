import os

import joblib
import numpy as np
import torch
from torch.utils import data


class ToyDataset(data.Dataset):

  def __init__(self, dataset_type='easy', partition_name='train', val_ratio= 0.05):

    dat_file = os.path.expanduser('~/data/stdnorm/{}.dat'.format(dataset_type))

    self.partition = {}
    if os.path.exists(dat_file):
      trX, trY, teX, teY = joblib.load(dat_file)

      self.trX = trX
      self.trY = trY
      self.teX = teX
      self.teY = teY

      self.val_ratio = val_ratio

      n_target_train = int(500 * self.val_ratio)

      self.X = np.vstack([self.trX, self.teX])
      self.y = np.hstack([self.trY, self.teY])

      self.partition_name = partition_name

      self.partition['train'] = np.hstack([np.arange(1000),
                                            np.arange(1000, 1000 + n_target_train),
                                            np.arange(1500, 1500 + n_target_train)])

      self.partition['test'] = np.hstack([np.arange(1000 + n_target_train, 1500),
                                            np.arange(1500 + n_target_train, 2000)])

      self.partition['validation'] = np.hstack([np.arange(1000, 1000+ n_target_train),
                                                np.arange(1500, 1500 + n_target_train)])

    else:
      raise Exception('Given dat file {} is missing.'.format(dat_file))

  def __getitem__(self, index):
    ID = self.partition[self.partition_name][index]
    return torch.tensor(self.X[ID], dtype=torch.float), torch.tensor(self.y[ID], dtype=torch.long)

  def __len__(self):
    return self.partition[self.partition_name].shape[0]
