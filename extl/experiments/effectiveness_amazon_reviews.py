import itertools
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from extl.dataset import load_data_set
from extl.models import get_classifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.contrib.learn.python.learn.datasets import base

from extl.influence.influence import BinaryLogisticRegressionWithLBFGS
from extl.influence.influence import DataSet

import hashlib
from extl.models.suba import SubspaceAlignedClassifier
from extl.models.tca import TransferComponentClassifier

all_results = []
all_data = []

domains = ['books', 'kitchen', 'electronics', 'dvd']

res = []
for d in itertools.permutations(domains, 2):

  source_domain = d[0]
  target_domain = d[1]

  print('***************************************************')
  print(' Source: {} and Target: {}'.format(source_domain, target_domain))

  data = load_data_set('multi-domain-sentiment', source=d[0], target=d[1])
  XS = data.XS
  XT = data.XT

  YS = data.yS
  YT = data.yT

  YS = YS * 0.5 + 0.5
  YT = YT * 0.5 + 0.5

  random_state = 0

  nTL = 50

  pos_inds = np.where(YT > 0.5)[0]
  neg_inds = np.where(YT < 0.5)[0]

  np.random.seed(random_state)
  np.random.shuffle(pos_inds)

  np.random.seed(random_state)
  np.random.shuffle(neg_inds)

  pos_inds_l = pos_inds[:nTL]
  neg_inds_l = neg_inds[:nTL]

  pos_inds_u = pos_inds[nTL:]
  neg_inds_u = neg_inds[nTL:]

  inds_l = np.hstack([pos_inds_l, neg_inds_l])
  inds_u = np.hstack([pos_inds_u, neg_inds_u])

  XS = np.vstack([XS, XT[inds_l, :]])
  XV = XT[inds_l, :]
  XT = XT[inds_u, :]


  YS = np.hstack([YS, YT[inds_l]])
  YV = YT[inds_l]
  YT = YT[inds_u]


  # Compute the influence.
  train = DataSet(XS, YS)
  validation = None
  test = DataSet(XT, YT)
  data_sets = base.Datasets(train=train, validation=validation, test=test)

  num_classes = 2

  input_dim = XS.shape[1]
  print(input_dim)
  weight_decay = 0.0001
  batch_size = 100
  initial_learning_rate = 0.001
  keep_probs = None
  decay_epochs = [1000, 10000]
  max_lbfgs_iter = 1000
  use_bias = True

  tf.reset_default_graph()

  orig_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='amazon_reviews_{}_{}'.format(d[0], d[1]))

  orig_model.train()

  seed = 0
  np.random.seed(seed)
  a = np.random.randint(0, 800, 50).tolist()
  b = np.random.randint(800, 1600, 50).tolist()

  test_idx = a + b

  md5 = hashlib.md5()
  md5.update(str(test_idx).encode('utf-8'))
  print(md5.hexdigest())

  test_description = 'amazon_reviews_{}_{}_{}'.format(md5.hexdigest(), d[0], d[1])

  num_train = len(orig_model.data_sets.train.labels)

  influences = orig_model.get_influence_on_test_loss(
    test_idx,
    np.arange(len(orig_model.data_sets.train.labels)),
    test_description=test_description,
    force_refresh=True) * num_train

  _inf = influences / np.max(np.abs(influences))
  _inf = _inf * 0.5 + 0.5
  _inf = _inf.reshape(-1, 1)
  _XS = np.multiply(XS, _inf)

  print('------------------------- EXPERIMENTS -------------------------------')

  # -------------------------------------------
  results = []

  clf = get_classifier('logistic')
  clf.fit(XS, YS)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = get_classifier('svm')
  clf.fit(XS, YS)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = SubspaceAlignedClassifier(num_components=1500, loss='logistic', l2=10)
  clf.fit(XS, YS, XT)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = TransferComponentClassifier(loss='logistic', mu=1., num_components=300, kernel_type='linear')
  clf.fit(XS, YS, XT)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = get_classifier('logistic')
  clf.fit(_XS, YS)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = get_classifier('svm')
  clf.fit(_XS, YS)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = SubspaceAlignedClassifier(num_components=1500, loss='logistic', l2=10)
  clf.fit(_XS, YS, XT)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = TransferComponentClassifier(loss='logistic', mu=1., num_components=300, kernel_type='linear')
  clf.fit(_XS, YS, XT)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  all_results.append(['{} > {}'.format(d[0], d[1])] + results)
  all_data.append([XS, _XS, _inf, XT, YS, YT, data.features])

  print(results)

  import joblib

  joblib.dump([XS, _XS, _inf, XT, YS, YT, data.features],
              '/home/arun/research/projects/tl-interpretability/code/intl/results/txt_amazon/data_{}_{}.dat'.format(
                d[0], d[1]), compress=3)

from tabulate import tabulate

print(tabulate(all_results,
               header=['Source > Target', 'LR', 'SVM', 'SUBA', 'TCA', 'INF LR', 'INF SVM', 'INF SUBA', 'INF-TCA']))
