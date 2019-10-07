import logging
import os

from scipy.io import loadmat
from torchvision.datasets.utils import download_url

from intl.dataset.base_dataset import BaseDataset


class SynthDigitsDataset(BaseDataset):

  def __init__(self, source=None, target=None):
    self.name = __name__

    self.logger = logging.getLogger(__name__)
    self.data_directory = os.path.join(os.getenv('DATA_DIR'), 'synth-digit')

    self.source = source
    self.target = target

    self.num_labels = 10
    self.image_shape = [16, 16, 1]

    self.file_names = {
      'train': 'synth_train_32x32_small.mat',
      'test': 'synth_test_32x32_small.mat'
    }

    if not os.path.exists(self.data_directory):
      self.logger.info('Synthetic digit data set is missing. Downloading from the source.')
      os.mkdir(self.data_directory)
      self._download_data_from_source(self.data_directory)

    self._load_dataset()

  def _download_data_from_source(self, root_dir):
    urls = {
      "train": "https://github.com/domainadaptation/datasets/blob/master/synth/synth_train_32x32_small.mat?raw=true",
      "test": "https://github.com/domainadaptation/datasets/blob/master/synth/synth_test_32x32_small.mat?raw=true"
    }

    for key in urls.keys():
      download_url(urls[key], self.data_directory, filename=self.file_names[key], md5=None)

  def _load_dataset(self):

    # training data.
    mat = loadmat(os.path.join(self.data_directory, self.file_names['train']))
    self.XS = mat['X'].transpose((3, 0, 1, 2))
    self.yS = mat['y'].squeeze()
    self.train_data = self.XS

    # test data
    mat = loadmat(os.path.join(self.data_directory, self.file_names['test']))
    self.XT = mat['X'].transpose((3, 0, 1, 2))
    self.yT = mat['y'].squeeze()
    self.test_data = self.XT


