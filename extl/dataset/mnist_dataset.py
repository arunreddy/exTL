import logging
import os
import joblib
import numpy as np

from torchvision.datasets import MNIST as MNISTBase

from intl.dataset.base_dataset import BaseDataset


class MNISTDataset(BaseDataset, MNISTBase):

  def __init__(self, source, target, **kwargs):
    self.name = __name__
    self.logger = logging.getLogger(__name__)
    self.data_directory = os.path.join(os.getenv('DATA_DIR'), 'mnist')

    self.source = source
    self.target = target


    if not os.path.exists(self.data_directory):
      self.logger.info('MNIST data directory is missing. Downloading from the source.')
      os.mkdir(self.data_directory)
      # self._download_data_from_source(self.data_directory)

    self._load_dataset()

  def _load_dataset(self):
    self.logger.info('Loading the MNIST dataset from root directory {}'.format(self.data_directory))
    MNISTBase.__init__(self, root=self.data_directory, download=True)

    XS = self.train_data.numpy()
    yS = self.train_labels.numpy()

    usps_data_path = 'usps/usps.dat'
    usps_data_path = os.path.join(os.getenv('DATA_DIR'), usps_data_path)
    XT, yT = joblib.load(usps_data_path)

    idx = []
    unique_classes = np.unique(yS)
    for _c in unique_classes:
      idx += np.where(yS == _c)[0][:700].tolist()

    _XS = XS[idx]
    _yS = yS[idx]

    idx = []
    unique_classes = np.unique(yT)
    for _c in unique_classes:
      idx += np.where(yT == _c)[0][:700].tolist()

    _XT = XT[idx]
    _yT = yT[idx]


    if self.source == 'mnist' and self.target == 'usps':
      self.XS = _XS
      self.yS = _yS
      self.XT = _XT
      self.yT = _yT
    elif self.source == 'usps' and self.target == 'mnist':
      self.XS = _XT
      self.yS = _yT
      self.XT = _XS
      self.yT = _yS
    else:
      raise  Exception('Unsupported source and target domains.')
    # idx = 10
    # label = 'Label {}'.format(self.train_labels[idx].numpy())
    # plot_image_from_data(self.train_dadta[idx], label)
