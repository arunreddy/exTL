import itertools
import os
import sys
import pandas as pd
import joblib
import numpy as np
from libtlda.iw import ImportanceWeightedClassifier
from libtlda.suba import SubspaceAlignedClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from libtlda.tca import TransferComponentClassifier
from tqdm import tqdm

from intl.tdata.sentiment_dataset_combined import SentimentDatasetCombined


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
    'random_state': random_state
  }

  params['partition'] = 'tr'
  tr_X, tr_y = get_data(SentimentDatasetCombined(**params))

  params['partition'] = 'te'
  te_X, te_y = get_data(SentimentDatasetCombined(**params))

  tr_y = tr_y.reshape(-1)
  te_y = te_y.reshape(-1)

  if model == 'lr':
    C = 0.2
    clf = LogisticRegression(solver='lbfgs', max_iter=1000, C=C)
    clf.fit(tr_X, tr_y)


  elif model == 'svm':
    C = 0.2
    clf = LinearSVC(C=C)
    clf.fit(tr_X, tr_y)

  elif model == 'kmm':
    clf = ImportanceWeightedClassifier(iwe='kmm')
    clf.fit(tr_X, tr_y, te_X)

  elif model == 'suba-lr':
    clf = SubspaceAlignedClassifier(loss='logistic')
    clf.fit(tr_X, tr_y, te_X)

  elif model == 'suba-hi':
    clf = SubspaceAlignedClassifier(loss='hinge')
    clf.fit(tr_X, tr_y, te_X)

  elif model == 'tca-lr':
    clf = TransferComponentClassifier(loss='logistic')
    clf.fit(tr_X, tr_y, te_X)

  elif model == 'tca-hi':
    clf = TransferComponentClassifier(loss='hinge')
    clf.fit(tr_X, tr_y, te_X)


  else:
    raise Exception('Unknown model called..')

  tr_score = accuracy_score(tr_y, clf.predict(tr_X))
  te_score = accuracy_score(te_y, clf.predict(te_X))

  return  tr_score, te_score


if __name__ == '__main__':

  target_train_ratio = 0.05

  RESULTS_DIR = os.path.expanduser('~/research/projects/tl-interpretability/results')


  if len(sys.argv) > 1:

    model = sys.argv[1]

    if len(sys.argv) > 2:
      target_train_ratio = float(sys.argv[2])

    domains = ['amazon', 'imdb', 'yelp']

    for source, target in itertools.permutations(domains, 2):


      l_tr_score = []
      l_te_score = []
      for i in tqdm(range(20), desc="[{}][{} {} {}][{}]".format(model, source[0], u'\u2192', target[0], target_train_ratio)):
        tr_score, te_score = main(source, target, model, target_train_ratio, random_state=i)
        l_tr_score.append(tr_score)
        l_te_score.append(te_score)

      cache_file = os.path.join(RESULTS_DIR, 'baselines', 'sentiment',
                                '{}_{}_{}_{:0.3f}.csv'.format(source, target, model, target_train_ratio))

      pd.DataFrame([l_tr_score, l_te_score]).to_csv(cache_file)

  else:
    print('Please enter the valid model name.')
