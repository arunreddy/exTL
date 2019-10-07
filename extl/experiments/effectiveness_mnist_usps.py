import hashlib
import itertools
import os

import tensorflow as tf
from sklearn.metrics import accuracy_score

from extl.dataset import load_data_set
from extl.models import get_classifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np

from extl.influence.influence import LogisticRegressionWithLBFGS
from extl.influence.influence import DataSet
from extl.models.suba import SubspaceAlignedClassifier

# Set up
domains = ['mnist', 'usps']
feat_type = 'surf'

res_all = []
for d in itertools.permutations(domains, 2):
  source_domain = d[0]
  target_domain = d[1]

  print('***************************************************')
  print(' Source: {} and Target: {}'.format(source_domain, target_domain))

  dataset = load_data_set('mnist', source=source_domain, target=target_domain, feat_type=feat_type)

  XS = dataset.XS.reshape(-1, 784)
  XT = dataset.XT.reshape(-1, 784)

  YS = dataset.yS
  YT = dataset.yT

  print(np.unique(YS), np.unique(YT))

  # TODO: normalize the images.
  print('Shape of the tdata (S and T)', XS.shape, XT.shape)

  min_value = min(min(XS.shape), min(XT.shape))
  print('Min value {}'.format(min_value))

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

  clf = SubspaceAlignedClassifier(num_components=min_value, loss='logistic', l2=0.1)
  clf.fit(XS, YS, XT)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  # clf = TransferComponentClassifier(loss='logistic', mu=1., num_components=300, kernel_type='linear')
  # clf.fit(XS, YS, XT)
  # yp = clf.predict(XT)
  # results.append(accuracy_score(yp, YT))

  # Train the influence
  # Compute the influence.
  train = DataSet(XS, YS)
  validation = None
  test = DataSet(XT, YT)
  data_sets = base.Datasets(train=train, validation=validation, test=test)

  num_classes = 10

  input_dim = XS.shape[1]
  weight_decay = 0.0001
  batch_size = 100
  initial_learning_rate = 0.001
  keep_probs = None
  decay_epochs = [1000, 10000]
  max_lbfgs_iter = 1000
  use_bias = True

  tf.reset_default_graph()

  orig_model = LogisticRegressionWithLBFGS(
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
    model_name='office_caltech_{}_{}'.format(d[0], d[1]))

  orig_model.train()

  seed = 0
  np.random.seed(seed)
  test_idx = np.random.randint(0, XT.shape[0], int(0.1 * XT.shape[0])).tolist()

  md5 = hashlib.md5()
  md5.update(str(test_idx).encode('utf-8'))
  print(md5.hexdigest())

  test_description = 'office-caltech{}_{}_{}'.format(md5.hexdigest(), d[0], d[1])

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

  clf = get_classifier('logistic')
  clf.fit(_XS, YS)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = get_classifier('svm')
  clf.fit(_XS, YS)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = SubspaceAlignedClassifier(num_components=min_value, loss='logistic', l2=10)
  clf.fit(_XS, YS, XT)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  # clf = TransferComponentClassifier(loss='logistic', mu=1., num_components=300, kernel_type='linear')
  # clf.fit(_XS, YS, XT)
  # yp = clf.predict(XT)
  # results.append(accuracy_score(yp, YT))

  res_all.append(['{} > {}'.format(d[0], d[1])] + results )

  print('----------------------------------------------------\n\n')


from tabulate import tabulate
header = ['Source > Target']
header.append('LR')
header.append('SVM')
header.append('SUBA')
# header.append('TCA')

header.append('INF_LR')
header.append('INF_SVM')
header.append('INF_SUBA')
# header.append('INF_TCA')

print(tabulate(res_all, headers=header))
