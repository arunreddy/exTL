import unittest

from extl.dataset.office_dataset import OfficeDataset


class OfficeDatasetTest(unittest.TestCase):

  def setUp(self):
    self.office_dataset = OfficeDataset()

  def test_load_domain(self):
    self.office_dataset.image_stats()
    X, y = self.office_dataset.load_domain('webcam', ['monitor'])
    self.assertEquals(X.shape, (43, 800))
    self.assertEquals(y.shape, (43,))
