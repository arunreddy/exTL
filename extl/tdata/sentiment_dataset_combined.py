import os

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from extl.tdata.base_dataset import BaseDataset
from extl.tdata.utils import shuffle_inplace


class SentimentDatasetCombined(BaseDataset):

  def __init__(self, source, target, partition, target_train_ratio=0.05, n_train=5000, max_features=1000,
               random_state=0,
               y_dtype='float'):
    self.source = source
    self.target = target
    self.max_features = max_features
    self.random_state = random_state
    self.target_train_ratio = target_train_ratio
    self.partition_dict = {}
    self.n_train = n_train

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

  def load_dataset(self):
    domains = ['amazon', 'imdb', 'yelp']
    data_dir = os.path.join(os.getenv('HOME'), 'data', 'sentiment')
    n = int(self.n_train / 2)
    n_target_train = int(n * self.target_train_ratio)
    cache_file = os.path.join(data_dir,
                              'combined_{}_{}_{}_{}.dat'.format(self.n_train, self.max_features, n_target_train,
                                                                self.random_state))

    if os.path.exists(cache_file):
      self.X, self.y, self.features, self.partition_dict = joblib.load(cache_file)

    else:
      data = []
      y = []
      for i, domain in enumerate(domains):
        pos, neg = joblib.load(os.path.join(data_dir, '{}.dat'.format(domain)))
        pos = pos[:n]
        neg = neg[:n]


        y.extend([1] * n + [0] * n)

        shuffle_inplace(pos, self.random_state)
        shuffle_inplace(neg, self.random_state)
        data.extend(pos + neg)

        inc = 2 * n * i
        tr_idx = np.arange(2 * n) + inc
        va_pos_idx = np.arange(0, n_target_train) + inc
        va_neg_idx = np.arange(n, n + n_target_train) + inc
        te_pos_idx = np.arange(n_target_train, n) + inc
        te_neg_idx = np.arange(n + n_target_train, 2 * n) + inc

        # set up the partition indices.
        self.partition_dict['{}_tr'.format(domain)] = np.hstack([tr_idx])

        self.partition_dict['{}_te'.format(domain)] = np.hstack([te_pos_idx,
                                                                 te_neg_idx])

        self.partition_dict['{}_va'.format(domain)] = np.hstack([va_pos_idx,
                                                                 va_neg_idx])

        count_vect = TfidfVectorizer(min_df=5,
                                     max_features=self.max_features,
                                     stop_words='english',
                                     strip_accents='unicode',
                                     ngram_range=(1, 3))

      X = count_vect.fit_transform(data)

      self.features = count_vect.get_feature_names()
      self.X = np.asarray(X.todense())

      self.y = np.asarray(y)

      joblib.dump([self.X, self.y, self.features, self.partition_dict], cache_file, compress=3)

  def __getitem__(self, index):
    ID = self.partition[index]
    return torch.tensor(self.X[ID], dtype=torch.float), torch.tensor(self.y[ID], dtype=self.y_dtype)

  def __len__(self):
    return self.partition.shape[0]


if __name__ == '__main__':
  tr_data = SentimentDatasetCombined(source='amazon', target='imdb', partition='tr', max_features=50, n_train=10,
                                     target_train_ratio=0.2)
