import os

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torchvision import transforms

from extl.tdata.base_dataset import BaseDataset
from extl.tdata.utils import xml_to_list, shuffle_inplace


class AmazonDataset(BaseDataset):

  def __init__(self, source, target, partition_name, target_train_ratio=0.05, max_features=1000, random_state=0, y_dtype='float'):
    self.max_features = max_features
    self.transform = transforms.Compose([transforms.ToTensor()])
    self.partition = {}
    self.partition_name = partition_name
    self.random_state = random_state
    self.target_train_ratio = target_train_ratio
    if y_dtype == 'long':
      self.y_dtype = torch.long
    else:
      self.y_dtype = torch.float

    super().__init__(source, target)

  def load_dataset(self):
    data_dir = os.path.join(os.getenv('HOME'), 'data', 'multi-domain-sentiment')

    cache_file = os.path.join(data_dir, '{}_{}.dat'.format(self.source, self.target))

    if os.path.exists(cache_file):
      src_pos, src_neg, tgt_pos, tgt_neg = joblib.load(cache_file)
    else:
      src_pos = xml_to_list(os.path.join(data_dir, self.source, 'positive.parsed'))
      src_neg = xml_to_list(os.path.join(data_dir, self.source, 'negative.parsed'))

      tgt_pos = xml_to_list(os.path.join(data_dir, self.target, 'positive.parsed'))
      tgt_neg = xml_to_list(os.path.join(data_dir, self.target, 'negative.parsed'))
      joblib.dump([src_pos, src_neg, tgt_pos, tgt_neg], cache_file, compress=3)

    # shuffle the list.
    shuffle_inplace(src_pos, self.random_state)
    shuffle_inplace(src_neg, self.random_state)
    shuffle_inplace(tgt_pos, self.random_state)
    shuffle_inplace(tgt_neg, self.random_state)

    data = src_pos + src_neg + tgt_pos + tgt_neg

    count_vect = TfidfVectorizer(min_df=5,
                                 max_features=self.max_features,
                                 stop_words='english',
                                 strip_accents='unicode',
                                 ngram_range=(1, 3))
    X = count_vect.fit_transform(data)
    y = np.asarray([1] * len(src_pos) + [0] * len(src_neg) + [1] * len(tgt_pos) + [0] * len(tgt_neg))

    self.features = count_vect.get_feature_names()

    self.X = np.asarray(X.todense())
    self.y = np.asarray(y)

    n_target_train = int(1000 * self.target_train_ratio)

    self.partition['source'] = np.hstack([np.arange(2000),
                                          np.arange(2000, 2000 + n_target_train),
                                          np.arange(3000, 3000 + n_target_train)])

    self.partition['target'] = np.hstack([np.arange(2000 + n_target_train, 3000),
                                          np.arange(3000 + n_target_train, 4000)])

    self.partition['validation'] = np.hstack([np.arange(2000, 2000 + n_target_train),
                                              np.arange(3000, 3000 + n_target_train)])

  def __getitem__(self, index):
    ID = self.partition[self.partition_name][index]
    return torch.tensor(self.X[ID], dtype=torch.float), torch.tensor(self.y[ID], dtype=self.y_dtype)

  def __len__(self):
    return len(self.partition[self.partition_name])
