import os
from glob import glob

import numpy as np
import scipy.io as sio


class OfficeDataset():

  def __init__(self, *args, **kwargs):

    super().__init__(*args, **kwargs)
    self.domains = ['amazon', 'dslr', 'webcam']
    self.categories = [
      'back_pack', 'bike_helmet', 'bottle', 'desk_chair',
      'desktop_computer', 'headphones', 'laptop_computer', 'mobile_phone',
      'mouse', 'paper_notebook', 'phone', 'projector',
      'ring_binder',
      'scissors', 'stapler', 'trash_can', 'bike', 'bookcase',
      'calculator', 'desk_lamp', 'file_cabinet', 'keyboard',
      'letter_tray', 'monitor', 'mug', 'pen', 'printer', 'punchers',
      'ruler', 'speaker', 'tape_dispenser']
    self.data_dir = os.path.join(
        os.getenv('DATASET_PATH'),
        'domain-adaptation-images')

  def image_stats(self):
    for domain in self.domains:
      print('{}'.format(domain))
      for category in self.categories:
        category_images_path = os.path.join(
            self.data_dir, 'images', domain, 'images', category)
        print("\t{}: {}".format(category, len(
            os.listdir(category_images_path))))

  def load_domain(self, domain, categories, feature_type='surf'):

    path = ''
    if feature_type == 'decaf':
      path = os.path.join(self.data_dir, 'decaf_features',
                          domain, 'interest_points')
    else:
      path = os.path.join(self.data_dir, 'surf_features',
                          domain, 'interest_points')

    files = []
    y = []
    for category in categories:
      feats_dir = os.path.join(path, category)
      print(feats_dir)
      for f in glob("{}/*.mat".format(feats_dir)):
        if 'amazon' in os.path.basename(f):
          files.append(f)
          y.append(category)

    X = []
    for f in files:
      x = sio.loadmat(f)['histogram']
      X.append(x)

    X = np.vstack(X)
    y = np.asarray(y)

    return X, y

  def load_source_target_domain(self):
    pass
