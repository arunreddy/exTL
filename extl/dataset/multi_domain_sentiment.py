import os

import numpy as np
import untangle
from sklearn.feature_extraction.text import TfidfVectorizer


from intl.dataset.base_dataset import BaseDataset


class MultiDomainSentiment2(BaseDataset):

  def __init__(self, source='dvd', target='electronics', **kwargs):
    self.data_dir = '/home/arun/research/projects/tl-interpretability/code/structural-correspondence-learning-SCL/data'
    self.source = source
    self.target = target

    self._load_dataset()

  def xml_to_list(self, file_name):
    doc = untangle.parse(file_name)
    reviews = []
    for i in range(len(doc.reviews)):
      txt = doc.reviews.review[i].cdata.strip()
      reviews.append(txt)

    return reviews

  def _load_dataset(self):
    src_pos = self.xml_to_list(os.path.join(self.data_dir, self.source, 'positive.parsed'))
    src_neg = self.xml_to_list(os.path.join(self.data_dir, self.source, 'negative.parsed'))

    tgt_pos = self.xml_to_list(os.path.join(self.data_dir, self.target, 'positive.parsed'))
    tgt_neg = self.xml_to_list(os.path.join(self.data_dir, self.target, 'negative.parsed'))

    data = src_pos + src_neg + tgt_pos + tgt_neg

    count_vect = TfidfVectorizer(min_df=10, max_features=4000, stop_words='english', ngram_range=[1,3])
    X = count_vect.fit_transform(data).toarray()

    self.features = count_vect.get_feature_names()

    X = X / np.max(X)

    XS = X[:2000]
    XT = X[2000:]

    yS = np.asarray([1] * 1000 + [-1] * 1000)
    yT = np.asarray([1] * 1000 + [-1] * 1000)


    self.XS = XS
    self.XT = XT
    self.yS = yS
    self.yT = yT

    self.dataS = src_pos + src_neg
    self.dataT = tgt_pos + tgt_neg
