import itertools
import os

import numpy as np
import tensorflow as tf

from extl.dataset import load_data_set
from extl.models import get_classifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.contrib.learn.python.learn.datasets import base

from extl.influence.influence import BinaryLogisticRegressionWithLBFGS
from extl.influence.influence import DataSet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import shap
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

  XS0 = np.vstack([XS, XT[inds_l, :]])
  XV0 = XT[inds_l, :]
  XT0 = XT[inds_u, :]


  YS0 = np.hstack([YS, YT[inds_l]])
  YV0 = YT[inds_l]
  YT0 = YT[inds_u]


  ###############################################################################################################
  ## STAGE - I

  clfA = LogisticRegression(solver='lbfgs')
  clfA.fit(XS0, YS0)
  YP0 = clfA.predict(XT0)
  score0 = accuracy_score(YT0, YP0)

  PP0 = clfA.predict_proba(XT0)
  entropy = np.multiply(-PP0, np.log(PP0)).sum(axis=1)
  test_idx = np.where(entropy >= 0.6)[0]

  num_classes = 2
  weight_decay = 0.0001
  batch_size = 100
  initial_learning_rate = 0.001
  keep_probs = None
  decay_epochs = [1000, 10000]
  max_lbfgs_iter = 1000
  use_bias = True

  # Compute the influence.
  train = DataSet(XS0, YS0)
  validation = None
  test = DataSet(XT0, YT0)
  data_sets = base.Datasets(train=train, validation=validation, test=test)
  input_dim = XS0.shape[1]

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
    model_name='amazon_reviews_{}_{}_0'.format(d[0], d[1]))
  orig_model.train()

  md5 = hashlib.md5()
  md5.update(str(test_idx).encode('utf-8'))

  test_description = 'amazon_reviews_{}_{}_{}_0'.format(md5.hexdigest(), d[0], d[1])

  num_train = len(orig_model.data_sets.train.labels)

  influences = orig_model.get_influence_on_test_loss(
    test_idx,
    np.arange(len(orig_model.data_sets.train.labels)),
    test_description=test_description,
    force_refresh=True) * num_train

  XS1 = np.multiply(XS0, MinMaxScaler().fit_transform(influences.reshape(-1,1)))

  clfB = LogisticRegression(solver='lbfgs')
  clfB.fit(XS1, YS0)
  score1 = accuracy_score(YT0, clfB.predict(XT0))

  explainer = shap.LinearExplainer(clfB, XS1, feature_dependence="independent")
  shap_values = explainer.shap_values(XS1)
  feat_shap_values = shap_values.sum(axis=0)

  abs_feat_values = np.abs(feat_shap_values)
  sorted_idx = np.argsort(-abs_feat_values)
  filt_idx = sorted_idx[:3500]

  XS2 = XS1[:, filt_idx]
  XT2 = XT0[:, filt_idx]


  ###############################################################################################################
  ## STAGE - II

  clfC = LogisticRegression(solver='lbfgs')
  clfC.fit(XS2, YS0)
  score2 = accuracy_score(YT0, clfC.predict(XT2))

  PP1 = clfC.predict_proba(XT2)
  entropy = np.multiply(-PP1, np.log(PP1)).sum(axis=1)
  test_idx = np.where(entropy >= 0.6)[0]

  # Compute the influence.
  train = DataSet(XS2, YS0)
  validation = None
  test = DataSet(XT2, YT0)
  data_sets = base.Datasets(train=train, validation=validation, test=test)
  input_dim = XS2.shape[1]

  tf.reset_default_graph()
  num_classes = 2
  weight_decay = 0.0001
  batch_size = 100
  initial_learning_rate = 0.001
  keep_probs = None
  decay_epochs = [1000, 10000]
  max_lbfgs_iter = 1000
  use_bias = True

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
    model_name='amazon_reviews_{}_{}_1'.format(d[0], d[1]))
  orig_model.train()

  md5 = hashlib.md5()
  md5.update(str(test_idx).encode('utf-8'))

  test_description = 'amazon_reviews_{}_{}_{}_1'.format(md5.hexdigest(), d[0], d[1])

  num_train = len(orig_model.data_sets.train.labels)

  influences = orig_model.get_influence_on_test_loss(
    test_idx,
    np.arange(len(orig_model.data_sets.train.labels)),
    test_description=test_description,
    force_refresh=True) * num_train

  XS3 = np.multiply(XS2, MinMaxScaler().fit_transform(influences.reshape(-1, 1)))

  clfD = LogisticRegression(solver='lbfgs')
  clfD.fit(XS3, YS0)
  score3 = accuracy_score(YT0, clfD.predict(XT2))

  explainer = shap.LinearExplainer(clfD, XS3, feature_dependence="independent")
  shap_values = explainer.shap_values(XS3)
  feat_shap_values = shap_values.sum(axis=0)

  abs_feat_values = np.abs(feat_shap_values)
  sorted_idx = np.argsort(-abs_feat_values)
  filt_idx = sorted_idx[:3250]

  XS4 = XS3[:, filt_idx]
  XT4 = XT2[:, filt_idx]

  ###############################################################################################################
  ## STAGE - III

  clfC = LogisticRegression(solver='lbfgs')
  clfC.fit(XS4, YS0)
  score4 = accuracy_score(YT0, clfC.predict(XT4))


  results = [score0, score1, score2, score3, score4]


  print('------------------------- EXPERIMENTS -------------------------------')

  clf = get_classifier('logistic')
  clf.fit(XS0, YS0)
  results.append(accuracy_score(clf.predict(XT0), YT0))

  clf = get_classifier('svm')
  clf.fit(XS0, YS0)
  results.append(accuracy_score(clf.predict(XT0), YT0))

  clf = SubspaceAlignedClassifier(num_components=1500, loss='logistic', l2=10)
  clf.fit(XS0, YS0, XT0)
  results.append(accuracy_score(clf.predict(XT0), YT0))

  clf = TransferComponentClassifier(loss='logistic', mu=1., num_components=300, kernel_type='linear')
  clf.fit(XS0, YS0, XT0)
  results.append(accuracy_score(clf.predict(XT0), YT0))


  clf = get_classifier('logistic')
  clf.fit(XS3, YS0)
  results.append(accuracy_score(clf.predict(XT2), YT0))

  clf = get_classifier('svm')
  clf.fit(XS3, YS0)
  results.append(accuracy_score(clf.predict(XT2), YT0))


  clf = SubspaceAlignedClassifier(num_components=1500, loss='logistic', l2=10)
  clf.fit(XS3, YS0, XT2)
  results.append(accuracy_score(clf.predict(XT2), YT0))

  clf = TransferComponentClassifier(loss='logistic', mu=1., num_components=300, kernel_type='linear')
  clf.fit(XS3, YS0, XT2)
  results.append(accuracy_score(clf.predict(XT2), YT0))


  all_results.append(['{} > {}'.format(d[0], d[1])] + results)
  # all_data.append([XS, _XS, _inf, XT, YS, YT, data.features])

  # joblib.dump([XS, _XS, _inf, XT, YS, YT, data.features],
  #             '/home/arun/research/projects/tl-interpretability/code/intl/results/txt_amazon/data_{}_{}.dat'.format(
  #               d[0], d[1]), compress=3)

#
#
# print(tabulate(all_results,
#                header=['Source > Target', 'LR', 'SVM', 'SUBA', 'TCA', 'INF LR', 'INF SVM', 'INF SUBA', 'INF-TCA']))
from tabulate import tabulate
print(tabulate(all_results))