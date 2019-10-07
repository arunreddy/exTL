import logging
import os

from torchvision.datasets import SVHN as SVHNBase

from intl.dataset.base_dataset import BaseDataset
from intl.dataset.utils import plot_image_from_data


class SVHNDataset(BaseDataset, SVHNBase):

  def __init__(self, **kwargs):
    self.name = __name__

    self.logger = logging.getLogger(__name__)
    self.data_directory = os.path.join(os.getenv('DATA_DIR'), 'svhn')

    if not os.path.exists(self.data_directory):
      self.logger.info('SVHN data directory is missing. Downloading from the source.')
      os.mkdir(self.data_directory)
      # self._download_data_from_source(self.data_directory)

    self._load_dataset()

  def _load_dataset(self):
    self.logger.info('Loading the SVHN dataset from root directory {}'.format(self.data_directory))
    SVHNBase.__init__(self, root=self.data_directory, download=True)

    self.XS = self.data
    self.yS = self.labels
    self.train_data = self.data.transpose((0, 2, 3 ,1))

    # idx = 10
    # label = 'Label {}'.format(self.labels[idx])
    # plot_image_from_data(self.data[idx, 1], label)
