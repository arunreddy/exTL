
from scipy.io import loadmat
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.contrib.learn.python.learn.datasets import base

from extl.influence.influence import LogisticRegressionWithLBFGS
from extl.influence.influence import DataSet

import hashlib

def run_amazon_dslr_experiments():


  data_dir = '/home/arun/Downloads/'
  # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
  domains = ['amazon.mat', 'dslr.mat', 'webcam.mat', ]

  for i in range(len(domains)):
    for j in range(len(domains)):
      if i != j:
        src, tar = data_dir + domains[i], data_dir + domains[j]
        src_domain, tar_domain = loadmat(src), loadmat(tar)
        Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']


        Xs = Xs.astype(np.float32)
        Xt = Xt.astype(np.float32)

        Ys = Ys.reshape(-1).astype(np.int32) - 1
        Yt = Yt.reshape(-1).astype(np.int32) - 1

        train = DataSet(Xs, Ys)
        validation = None
        test = DataSet(Xt, Yt)
        data_sets = base.Datasets(train=train, validation=validation, test=test)

        input_dim = Xs.shape[1]
        weight_decay = 0.01
        batch_size = 100
        initial_learning_rate = 0.001
        keep_probs = None
        decay_epochs = [1000, 10000]
        max_lbfgs_iter = 1000
        num_classes = 10

        tf.reset_default_graph()

        model = LogisticRegressionWithLBFGS(
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
            train_dir='data',
            log_dir='log',
            model_name='amazfdson_images_{}_{}'.format(i,j))

        model.train(verbose=False)


        # compute influence scores..
        np.random.seed(0)
        test_idx = np.arange(Yt.size) #np.random.randint(0,Yt.size,50)

        md5 = hashlib.md5()
        md5.update(str(test_idx).encode('utf-8'))
        print(md5.hexdigest())

        test_description = '{}_{}_{}'.format(md5.hexdigest(),i,j)

        num_train = len(model.data_sets.train.labels)

        influences = model.get_influence_on_test_loss(
            test_idx,
            np.arange(len(model.data_sets.train.labels)),
            test_description=test_description,
            force_refresh=True) * num_train

        _inf = influences/np.max(np.abs(influences))
        _inf = _inf*0.5 + 0.5
        _inf = _inf.reshape(-1, 1)
        _Xs = np.multiply(Xs, _inf)

        train = DataSet(Xs, Ys)
        validation = None
        test = DataSet(Xt, Yt)
        data_sets = base.Datasets(train=train, validation=validation, test=test)

        tf.reset_default_graph()

        new_model = LogisticRegressionWithLBFGS(
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
          train_dir='data',
          log_dir='log',
          model_name='new_amazon_images_{}_{}'.format(i, j))

        new_model.train(verbose=False)


        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        clf.fit(Xs, Ys.reshape(-1))
        yP = clf.predict(Xt)
        print(domains[i][0], domains[j][0], accuracy_score(yP, Yt.reshape(-1)))

        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        clf.fit(_Xs, Ys.reshape(-1))
        yP = clf.predict(Xt)
        print(domains[i][0], domains[j][0], accuracy_score(yP, Yt.reshape(-1)))
        print('------------------------------------------------------------------')


