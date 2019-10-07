import os

import joblib
import numpy as np
import scipy.io as sio
import torch
from extl.tdata.base_dataset import BaseDataset


class OfficeCaltechCombined(BaseDataset):

  def __init__(self, source, target, partition, target_train_ratio=0.1, feat_type='surf', random_state=0,
               y_dtype='float'):

    self.data_dir = os.path.join(os.getenv('HOME'), 'data', 'OfficeCaltechDomainAdaptation')
    self.feat_type = feat_type
    self.source = source
    self.target = target
    self.random_state = random_state
    self.target_train_ratio = target_train_ratio
    self.partition_dict = {}

    self.load_dataset()

    if y_dtype == 'long':
      self.y_dtype = torch.long
    else:
      self.y_dtype = torch.float

    if partition == 'tr':
      self.partition = np.hstack(
          [self.partition_dict['{}_tr'.format(source)], self.partition_dict['{}_va'.format(target)]])
    elif partition == 'te':
      self.partition = self.partition_dict['{}_te'.format(target)]
    elif partition == 'va':
      self.partition = self.partition_dict['{}_va'.format(target)]

  def _load_domain_information(self, domain):
    d = sio.loadmat(os.path.join(self.data_dir, 'features/{}/{}.mat'.format(self.feat_type, domain)))
    X = d['fts']
    y = d['labels']
    if y.shape[0] == 1:
      y = y.transpose().reshape(-1)
    else:
      y = y.reshape(-1)
    return X, y

  def _split_data(self, labels):
    train = []
    val = []
    test = []

    for label in np.unique(labels):
      label_idx = np.where(labels == label)[0]

      # shuffle in place
      np.random.seed(self.random_state)
      np.random.shuffle(label_idx)

      train.extend(label_idx.tolist())

      n_va = int(self.target_train_ratio * label_idx.shape[0])
      val.extend(label_idx[:n_va].tolist())
      test.extend(label_idx[n_va:].tolist())

    tr_idx = np.asarray(train)
    va_idx = np.asarray(val)
    te_idx = np.asarray(test)

    return tr_idx, va_idx, te_idx

  def load_dataset(self):

    self.domains = ['amazon', 'caltech10', 'dslr', 'webcam']

    cache_file = os.path.join(self.data_dir, 'combined_{}_{}_{}.dat'.format(self.target_train_ratio,
                                                                            self.feat_type,
                                                                            self.random_state))

    if os.path.exists(cache_file):
      self.X, self.y,  self.partition_dict = joblib.load(cache_file)
    else:
      X = []
      y = []

      inc = 0
      for domain in self.domains:
        _X, _y = self._load_domain_information(domain)
        tr_idx, va_idx, te_idx = self._split_data(_y)

        self.partition_dict['{}_tr'.format(domain)] = tr_idx + inc
        self.partition_dict['{}_te'.format(domain)] = te_idx + inc
        self.partition_dict['{}_va'.format(domain)] = va_idx + inc

        inc += _y.shape[0]

        if np.min(_y) == 1:
          _y -= 1

        X.append(_X)
        y.append(_y)

      self.X = np.vstack(X)
      self.y = np.hstack(y)
      joblib.dump([self.X, self.y, self.partition_dict], cache_file, compress=3)
