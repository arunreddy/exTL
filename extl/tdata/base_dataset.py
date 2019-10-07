from torch.utils import data
from abc import ABC, abstractmethod
from abc import ABC, abstractmethod

from torch.utils import data


class BaseDataset(data.Dataset, ABC):

  def __init__(self, source, target):

    self.source = source
    self.target = target
    self.X = None
    self.y = None
    self.features = None

    self.load_dataset()

  # Use TextBlob


  @abstractmethod
  def load_dataset(self):
    pass
  

  def __len__(self):
    pass


  def __getitem__(self, item):
    pass