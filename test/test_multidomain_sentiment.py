import unittest

from extl.dataset.multi_domain_sentiment_dataset import MultiDomainSentiment


class MultiDomainSentimentTest(unittest.TestCase):

  def setUp(self):
    self.mds = MultiDomainSentiment()

  def test_domain_with_stopwords(self):
    X, y, features = self.mds.load_data('books', filter_stopwords=False)
    self.assertEqual(y.shape[0], 2000)
    self.assertEqual(X.shape[0], 2000)
    self.assertEqual(X.shape[1], 195887)

  def test_domain(self):
    X, y, features = self.mds.load_data('books', filter_stopwords=True)
    self.assertEqual(y.shape[0], 2000)
    self.assertEqual(X.shape[0], 2000)
    self.assertEqual(X.shape[1], 195640)

  def test_source_target(self):
    X_S, y_S, X_T, y_T, features = self.mds.load_source_target_data(
      'books', 'dvd')
    self.assertEqual(X_S.shape[0], 2000)
    self.assertEqual(y_S.shape[0], 2000)
    self.assertEqual(X_T.shape[0], 2000)
    self.assertEqual(y_T.shape[0], 2000)
