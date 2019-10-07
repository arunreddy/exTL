from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import vstack
from scipy.sparse.csc import csc_matrix
from sklearn.manifold import TSNE
import matplotlib.image as mpimg

class BaseDataset(ABC):

  def __init__(self, **kwargs):

    self.XS = None
    self.yS = None
    self.XT = None
    self.yT = None

    self.dataS = None
    self.dataT = None

    self.m = None
    self.n = None

    self.features = None

    self.name = None

    # load configuration from kwargs
    self.random_state = kwargs.get('random_state', None)

    self._load_dataset()

  @abstractmethod
  def _load_dataset(self):
    raise NotImplementedError

  def plot_data(self):
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

    if type(self.XS) is csc_matrix:
      X_CSC = vstack((self.XS, self.XT))
      X = X_CSC.toarray()
    else:
      X = np.vstack([self.XS, self.XT])

    X_2D = TSNE(n_components=2, perplexity=50, random_state=4).fit_transform(X)
    c = cS + cT

    plt.figure(figsize=(10, 20))
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=c)
    plt.show()


  def plot_image(self,n_img=16):

    fig = plt.figure(figsize=(20, 20))
    plt.title('Examples from dataset {}'.format(self.name))
    n_rows = 4
    n_cols = 4
    for i in range(n_img):
      fig.add_subplot(n_rows, n_cols, i + 1)
      plt.imshow(self.train_data[i])
      plt.title('Image - {}'.format(self.yS[i]))


    plt.show()
