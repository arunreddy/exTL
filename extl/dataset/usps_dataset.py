'''
Adapted from https://github.com/domainadaptation/salad/blob/master/salad/datasets/digits/usps.py
'''
import logging
import os

import numpy as np
from torchvision.datasets.utils import download_url

from intl.dataset.base_dataset import BaseDataset


class USPSDataset(BaseDataset):

  def __init__(self, source=None, target=None):

    self.logger = logging.getLogger(__name__)
    self.data_directory = os.path.join(os.getenv('DATA_DIR'), 'usps')
    self.name = __name__

    self.source = source
    self.target = target

    self.num_labels = 10
    self.image_shape = [16, 16, 1]

    self.file_names = {
      'train': 'zip.train.gz',
      'test': 'zip.test.gz'
    }

    if not os.path.exists(self.data_directory):
      self.logger.info('Synthetic digit data set is missing. Downloading from the source.')
      os.mkdir(self.data_directory)
      self._download_data_from_source(self.data_directory)

    self._load_dataset()

  def _download_data_from_source(self, root_dir):
    urls = {
      'train': 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz',
      'test': 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz'
    }

    for key in urls.keys():
      download_url(urls[key], self.data_directory, filename=self.file_names[key], md5=None)

  def _extract_images_labels(self, filename):

    import gzip

    with gzip.open(filename, 'rb') as f:
      raw_data = f.read().split()
    data = np.asarray([raw_data[start:start + 257]
                       for start in range(0, len(raw_data), 257)],
                      dtype=np.float32)
    images_vec = data[:, 1:]
    images = np.reshape(images_vec, (images_vec.shape[0], 16, 16))
    labels = data[:, 0].astype(int)
    images = ((images + 1) * 128).astype('uint8')

    return images, labels

  def _load_dataset(self):

    self.XS, self.yS = self._extract_images_labels(os.path.join(self.data_directory, self.file_names['train']))
    self.XT, self.yT = self._extract_images_labels(os.path.join(self.data_directory, self.file_names['test']))

    self.train_data = self.XS
    self.test_data = self.XT

    # idx = np.random.randint(0,self.XS.shape[0],1)[0]
    # label = 'Label {}'.format(self.yS[idx])
    # plot_image_from_data(self.XS[idx], label)
