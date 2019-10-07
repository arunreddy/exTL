import os
import numpy as np
import scipy.io as sio

from sklearn.feature_extraction.text import TfidfVectorizer

from intl.dataset.base_dataset import BaseDataset


class OfficeCaltechDataset(BaseDataset):

  def __init__(self, source='dvd', target='electronics', feat_type='surf', **kwargs):
    self.data_dir = '/home/arun/research/projects/tl-interpretability/data/OfficeCaltechDomainAdaptation/'
    self.source = source
    self.target = target


    if feat_type == 'images':
      self.feat_type = 'images'
    elif feat_type == 'google':
      self.feat_type = 'GoogleNet1024' 
    elif feat_type == 'caffe':
      self.feat_type = 'CaffeNet4096' 
    else:
      self.feat_type = 'surf' 


    self._load_dataset()

  def _load_domain_information(self, domain):
    d = sio.loadmat(os.path.join(self.data_dir, 'features/{}/{}.mat'.format(self.feat_type, domain)))

    X = d['fts']

    y = d['labels']
    if y.shape[0] == 1:
        y = y.transpose().reshape(-1)
    else:
        y = y.reshape(-1)


    return X, y

  def _load_dataset(self):


    if self.feat_type != 'images':
      self.XS, self.yS = self._load_domain_information(self.source)
      self.XT, self.yT = self._load_domain_information(self.target)

    else:
      pass
