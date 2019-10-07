import os
import joblib
import numpy as np

from intl.dataset.base_dataset import BaseDataset
from sklearn.feature_extraction.text import TfidfVectorizer

class SentimentDataset(BaseDataset):

  def __init__(self, source, target, n_top = -1,**kwargs):
    self.name = __name__
    self.source = source
    self.target = target
    self.data_dir = '/home/arun/research/projects/tl-interpretability/data/sentiment'
    self.n_top = n_top
    self._load_dataset()

  def _load_dataset(self):
    A = joblib.load(os.path.join(self.data_dir, '{}.dat'.format(self.source)))
    src_pos, src_neg = joblib.load(os.path.join(self.data_dir, '{}.dat'.format(self.source)))
    tgt_pos, tgt_neg = joblib.load(os.path.join(self.data_dir, '{}.dat'.format(self.target)))

    if self.n_top > 0:
      data = src_pos[:self.n_top] + src_neg[:self.n_top] + tgt_pos[:self.n_top] + tgt_neg[:self.n_top]
      yS = np.asarray([1] * self.n_top + [-1] * self.n_top)
      yT = np.asarray([1] * self.n_top + [-1] * self.n_top)
      nS = 2*self.n_top

    else:
      data = src_pos + src_neg + tgt_pos + tgt_neg
      yS = np.asarray([1] * len(src_pos) + [-1] * len(src_neg))
      yT = np.asarray([1] * len(tgt_pos) + [-1] * len(tgt_neg))
      nS = len(src_pos) + len(src_neg)

    count_vect = TfidfVectorizer(min_df=10, max_features=4000, stop_words='english', ngram_range=[1,2])
    X = count_vect.fit_transform(data).toarray()

    self.features = count_vect.get_feature_names()

    X = X / np.max(X)


    XS = X[:nS]
    XT = X[nS:]

    self.XS = XS
    self.XT = XT
    self.yS = yS
    self.yT = yT

    self.dataS = src_pos + src_neg
    self.dataT = tgt_pos + tgt_neg
