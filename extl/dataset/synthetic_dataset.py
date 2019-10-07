import logging

import matplotlib.pyplot as plt
import numpy  as np
import numpy.random as rnd
from sklearn.manifold import TSNE

from .base_dataset import BaseDataset


class SyntheticDataset(BaseDataset):

  def __init__(self, random_state=0, **kwargs):

    self.logger = logging.getLogger(__name__)
    # load configuration
    self.nS = kwargs.get('nS', 1000)
    self.nT = kwargs.get('nT', 1000)
    self.nDim = kwargs.get('nDim', 20)

    self.labels = [-1, 1]

    self.random_state = random_state
    self._load_dataset()

    # super(SyntheticDataset, self).__init__(**kwargs)

  def _load_dataset(self):

    self.logger.info('Loading the synthetic dataset.')

    cmn_inds = np.arange(10)#np.random.randint(0,10,8)
    src_inds = np.arange(10,14) #np.random.randint(10,20,8)
    tgt_inds = np.arange(14,20) #np.random.randint(20, 30, 8)


    # Source domain data.
    sigma_S = 2.5
    mu_S0 = np.ones(self.nDim)

    for idx in cmn_inds:
      mu_S0[idx] = 2

    for idx in src_inds:
      mu_S0[idx] = 2


    mu_S1 = np.ones(self.nDim)
    for idx in cmn_inds:
      mu_S1[idx] = -2

    for idx in src_inds:
      mu_S1[idx] = -1


    M0 = int(self.nS / 2)
    M1 = self.nS - M0

    np.random.seed(self.random_state)
    X0 = rnd.randn(M0, self.nDim) * sigma_S + mu_S0

    np.random.seed(self.random_state+100)
    X1 = rnd.randn(M1, self.nDim) * sigma_S + mu_S1
    self.XS = np.concatenate((X0, X1), axis=0)
    self.yS = np.concatenate((self.labels[0] * np.ones((M0,), dtype='int'),
                              self.labels[1] * np.ones((M1,), dtype='int')), axis=0)

    # Target domain data.
    sigma_T = 3.5
    mu_T0 = np.ones(self.nDim)
    for idx in cmn_inds:
      mu_T0[idx] = 1

    for idx in tgt_inds:
      mu_T0[idx] = 1

    mu_T1 = np.ones(self.nDim)
    for idx in cmn_inds:
      mu_T1[idx] = -3.

    for idx in tgt_inds:
      mu_T1[idx] = -4

    N0 = int(self.nT / 2)
    N1 = self.nT - N0

    np.random.seed(self.random_state + 200)
    X0 = rnd.randn(N0, self.nDim) * sigma_T + mu_T0
    np.random.seed(self.random_state + 300)
    X1 = rnd.randn(N1, self.nDim) * sigma_T + mu_T1
    self.XT = np.concatenate((X0, X1), axis=0)
    self.yT = np.concatenate((self.labels[0] * np.ones((N0,), dtype='int'),
                              self.labels[1] * np.ones((N1,), dtype='int')), axis=0)

  def plot_data(self, savefig = True):

    cS = []
    for _y in self.yS:
      if _y > 0:
        cS.append('b')
      else:
        cS.append('r')

    cT = []
    for _y in self.yT:
      if _y > 0:
        cS.append('g')
      else:
        cS.append('m')

    X = np.vstack([self.XS, self.XT])
    clf = TSNE(n_components=2, perplexity=100)
    X_2D = clf.fit_transform(X)
    c = cS + cT

    plt.figure(figsize=(10, 10))
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=c)

    if savefig:
      plt.savefig('results/synthetic/results_{}.png'.format(self.random_state))
    else:
      plt.show()

    return clf
