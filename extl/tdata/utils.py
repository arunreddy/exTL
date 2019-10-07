import numpy as np
import untangle


def xml_to_list(file_name):
  doc = untangle.parse(file_name)
  reviews = []
  for i in range(len(doc.reviews)):
    txt = doc.reviews.review[i].cdata.strip()
    reviews.append(txt)

  return reviews


def shuffle_inplace(l, random_state=0):
  np.random.seed(random_state)
  np.random.shuffle(l)


if __name__ == '__main__':
  l = np.arange(10)
  print(l)

  l = np.arange(10)
  shuffle_inplace(l, 0)
  print(l)

  l = np.arange(10)
  shuffle_inplace(l, 1)
  print(l)

  l = np.arange(10)
  shuffle_inplace(l, 2)
  print(l)
