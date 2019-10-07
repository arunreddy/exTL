import logging
import os
import tarfile

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import vstack
from spacy.lang.en.stop_words import STOP_WORDS
from torchvision.datasets.utils import download_url

from intl.dataset.base_dataset import BaseDataset


class MultiDomainSentiment(BaseDataset):

  def __init__(self, source='dvd', target='electronics', **kwargs):

    self.logger = logging.getLogger(__name__)
    self.data_directory = os.path.join(os.getenv('DATA_DIR'), 'multi-domain-sentiment')

    self.source = source
    self.target = target

    if not os.path.exists(self.data_directory):
      self.logger.info('Multi domain sentiment data set is missing. Downloading from the source.')
      os.mkdir(self.data_directory)
      self._download_data_from_source(self.data_directory)

    self.processed_acl_file = os.path.join(self.data_directory, 'processed_acl.tar.gz')
    if not os.path.exists(self.processed_acl_file):
      self._download_data_from_source(self.data_directory)

    self.domains = ['books', 'kitchen', 'dvd', 'electronics']

    self._load_dataset()

  def _download_data_from_source(self, root_dir):
    url = 'https://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_acl.tar.gz'
    download_url(url, self.data_directory, filename='processed_acl.tar.gz', md5=None)

  def _load_dataset(self):
    XS, yS, XT, yT, features = self.load_source_target_data(self.source, self.target)

    # filter non-frequent features.
    X = vstack((XS, XT))
    X_SUM = X.sum(axis=0).transpose()
    indices = np.where(X_SUM > 20)[0]

    self.XS = XS[:, indices]
    self.yS = yS
    self.XT = XT[:, indices]
    self.yT = yT
    self.features = [features[i] for i in indices]
    self.m = XS.shape[0]
    self.n = XT.shape[0]

  def get_processed_data(self, domain):
    """Fetch the dataset for the given domain.

    Arguments:
      domain {[type]} -- [description]
    """
    if domain not in self.domains:
      raise Exception(
          'Provided domain name {} doesn\'t exist in the dataset'.format(
              domain))

    if not os.path.exists(self.processed_acl_file):
      raise Exception(
          'The dataset file {} is not found.'.format(
              self.processed_acl_file))

    with tarfile.open(self.processed_acl_file, 'r') as f:
      neg_file = "processed_acl/{}/negative.review".format(domain)
      neg_content = f.extractfile(neg_file).read().split(b'\n')[:1000]

      pos_file = "processed_acl/{}/positive.review".format(domain)
      pos_content = f.extractfile(pos_file).read().split(b'\n')[:1000]
      data = neg_content + pos_content
      return data

    return None

  def extract_features_labels_from_data(self, data, filter_stopwords=True):
    if data is None:
      return None

    vocabulary = dict()
    vocab_index = 0
    r = []
    c = []
    v = []
    labels = []

    for idx in range(len(data)):
      line = data[idx].decode('utf-8')
      features = line.split(' ')
      for f_idx in range(len(features) - 1):
        feature = features[f_idx]
        word, freq = feature.split(':')

        if filter_stopwords and word in STOP_WORDS:
          continue

        if word not in vocabulary:
          vocabulary[word] = vocab_index
          vocab_index += 1

        word_idx = vocabulary.get(word)
        r.append(idx)
        c.append(word_idx)
        v.append(int(freq))

      label = features[-1].split(':')[1]
      labels.append(label)

    r = np.asarray(r)
    c = np.asarray(c)
    v = np.asarray(v)

    X = sparse.csc_matrix((v, (r, c)))
    y = []
    for label in labels:
      if label == 'positive':
        y.append(1)
      else:
        y.append(-1)
    y = np.asarray(y)
    features = list(vocabulary.keys())

    return X, y, features

  def load_data(self, domain, filter_stopwords=True):
    data = self.get_processed_data(domain)
    X, y, features = self.extract_features_labels_from_data(
        data, filter_stopwords)
    return X, y, features

  def load_source_target_data(
      self, source_domain, target_domain, filter_stopwords=True):
    """Return the data from given source and target domain.

    Arguments:
      source_domain {str} -- name of the source domain
      target_domain {str} -- name of the target domain
    """
    data_S = self.get_processed_data(source_domain)
    data_T = self.get_processed_data(target_domain)
    m = len(data_S)

    data = data_S + data_T
    X, y, features = self.extract_features_labels_from_data(
        data, filter_stopwords)
    X_S = X[:m, :]
    X_T = X[m:, :]
    y_S = y[:m]
    y_T = y[m:]
    return X_S, y_S, X_T, y_T, features
