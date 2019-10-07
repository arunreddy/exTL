import itertools
import os
import sys

import numpy as np
import pandas as pd
from libtlda.iw import ImportanceWeightedClassifier
from libtlda.suba import SubspaceAlignedClassifier
from libtlda.tca import TransferComponentClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from intl.models.iw.mlp import MinibatchMLP
from intl.tdata.amazon_dataset_combined import AmazonDatasetCombined


def get_data(dataset):
  lX = []
  ly = []
  for i in range(len(dataset)):
    x, y = dataset.__getitem__(i)
    lX.append(x)
    ly.append(y)

  X = np.vstack(lX)
  y = np.vstack(ly)

  return X, y


def main(source, target, model, target_train_ratio, random_state):
  params = {
    'source': source,
    'target': target,
    'target_train_ratio': target_train_ratio,
    'max_features': 5000,
    'random_state': random_state,
    'y_dtype': 'long',
    'device': 'cpu'
  }

  params['partition'] = 'tr'
  tr_data = AmazonDatasetCombined(**params)

  params['partition'] = 'te'
  params['target_train_ratio'] = 0.1
  te_data = AmazonDatasetCombined(**params)

  params['partition'] = 'va'
  params['target_train_ratio'] = 0.1
  va_data = AmazonDatasetCombined(**params)

  batch_size = 50

  tr_loader = DataLoader(tr_data, batch_size=100, shuffle=True)
  te_loader = DataLoader(te_data, batch_size=100, shuffle=True)
  # va_loader = DataLoader(va_data, batch_size=50, shuffle=True)

  iw = MinibatchMLP(tr_loader,
                    te_loader,
                    None,
                    input_dim=5000)

  data = iw.train()

  return data


if __name__ == '__main__':

  target_train_ratio = 0.05

  RESULTS_DIR = os.path.expanduser('~/research/projects/tl-interpretability/results')

  if len(sys.argv) > 1:
    target_train_ratio = float(sys.argv[1])

  model = 'minibatch_mlp'

  domains = ['books', 'dvd', 'electronics', 'kitchen']

  for source, target in itertools.permutations(domains, 2):

    data = []
    for i in tqdm(range(20),
                  desc="[{}][{} {} {}][{}]".format(model, source[0], u'\u2192', target[0], target_train_ratio)):
      _data = main(source, target, model, target_train_ratio, random_state=i)
      data.append(_data)

    cache_file = os.path.join(RESULTS_DIR, 'baselines', 'multi-domain-sentiment',
                              '{}_{}_{}_{:0.3f}.csv'.format(source, target, model, target_train_ratio))

    pd.DataFrame(data).to_csv(cache_file)

  else:
    print('Please enter the valid model name.')
