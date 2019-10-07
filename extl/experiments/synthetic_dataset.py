from intl.dataset import load_data_set
from intl.models import get_classifier

from sklearn.metrics import accuracy_score
from intl.models.suba import SubspaceAlignedClassifier
from intl.models.tca import TransferComponentClassifier

import os

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.contrib.learn.python.learn.datasets import base

from intl.influence.influence import BinaryLogisticRegressionWithLBFGS
from intl.influence.influence import DataSet
import numpy as np
import hashlib

def run_synthetic_experiments(random_state):
  data = load_data_set('synthetic', random_state=random_state)

  tsne =  data.plot_data(savefig=True)

  XS = data.XS
  XT = data.XT

  YS = data.yS
  YT = data.yT

  YS = YS * 0.5 + 0.5
  YT = YT * 0.5 + 0.5

  # Compute the influence.
  train = DataSet(XS, YS)
  validation = None
  test = DataSet(XT, YT)
  data_sets = base.Datasets(train=train, validation=validation, test=test)

  num_classes = 2

  input_dim = XS.shape[1]
  print(input_dim)
  weight_decay = 0.0001
  batch_size = 20
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
    model_name='synth_dataset')

  orig_model.train()

  seed = 0
  np.random.seed(seed)
  a = np.random.randint(0, 500, 50).tolist()
  np.random.seed(seed)
  b = np.random.randint(500, 1000, 50).tolist()

  test_idx = a + b

  md5 = hashlib.md5()
  md5.update(str(test_idx).encode('utf-8'))
  print(md5.hexdigest())

  test_description = 'synth_{}'.format(md5.hexdigest())

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

  results = []


  clf = get_classifier('logistic')
  clf.fit(XS, YS)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = get_classifier('svm')
  clf.fit(XS, YS)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = SubspaceAlignedClassifier(num_components=XS.shape[1], loss='hinge', l2=0.1)
  clf.fit(XS, YS, XT)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = TransferComponentClassifier(loss='logistic', mu=1., num_components=40, kernel_type='rbf')
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

  clf = SubspaceAlignedClassifier(num_components=XS.shape[1], loss='hinge', l2=0.1)
  clf.fit(_XS, YS, XT)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))

  clf = TransferComponentClassifier(loss='logistic', mu=1., num_components=40, kernel_type='rbf')
  clf.fit(_XS, YS, XT)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, YT))



  return results, [XS, _XS, _inf, XT, YS, YT, tsne]
