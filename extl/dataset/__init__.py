from extl.dataset.mnist_dataset import MNISTDataset
from extl.dataset.multi_domain_sentiment import MultiDomainSentiment2
from extl.dataset.sentiment_dataset import SentimentDataset
from extl.dataset.multi_domain_sentiment_dataset import MultiDomainSentiment
from extl.dataset.svhn_dataset import SVHNDataset
from extl.dataset.synth_digits_dataset import SynthDigitsDataset
from extl.dataset.synthetic_dataset import SyntheticDataset
from extl.dataset.usps_dataset import USPSDataset
from extl.dataset.office_caltech_dataset import OfficeCaltechDataset

def load_data_set(name='', source=None, target=None, random_state=0, feat_type='surf', n_top=-1, **kwargs):
  data_set = None

  if name == 'synthetic':
    data_set = SyntheticDataset(random_state=random_state)
    data_set._load_dataset()  # reload the dataset.

  elif name == 'sentiment':
    data_set = SentimentDataset(source=source, target=target, n_top=n_top, **kwargs)

  elif name == 'multi-domain-sentiment':
    data_set = MultiDomainSentiment2(source=source, target=target, **kwargs)

  elif name == 'mnist':
    data_set = MNISTDataset(source=source, target=target, **kwargs)

  elif name == 'svhn':
    data_set = SVHNDataset()

  elif name == 'synth-digit':
    data_set = SynthDigitsDataset()

  elif name == 'usps':
    data_set = USPSDataset()

  elif name == 'office-caltech':
    data_set = OfficeCaltechDataset(source=source, target=target, feat_type=feat_type)

  else:
    print('The given data set is not supported.')
    raise Exception('Given data set {} is not supported.'.format(name))

  return data_set


if __name__ == '__main__':
  import os
  import numpy as np

  os.environ['DATA_DIR'] = '/home/arun/research/projects/tl-interpretability/data'
  os.environ['RESULTS_DIR'] = '/home/arun/research/projects/tl-interpretability/code/intl/results'
  dataset = load_data_set('mnist')
  print(dataset.train_data.shape)
  # dataset.plot_image()
  import joblib
  joblib.dump([dataset.train_data, dataset.yS],'/tmp/mnist.dat', compress=3)

  #
  # dataset = load_data_set('svhn')
  # print(dataset.train_data.shape)
  # print(np.unique(dataset.yS))
  # dataset.plot_image()
  #
  #
  # dataset = load_data_set('synth-digit')
  # print(dataset.train_data.shape)
  # dataset.plot_image()

  # dataset.plot_data(savefig=False)
  dataset = load_data_set('usps')
  print(dataset.train_data.shape)

  x = dataset.train_data[0]
  print(x)


  joblib.dump([dataset.train_data, dataset.yS],'/tmp/svhn.dat', compress=3)

  # dataset.plot_image()
