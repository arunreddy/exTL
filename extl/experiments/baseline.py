from extl.dataset import load_data_set
from extl.models import get_classifier
import scipy.sparse as sp
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.preprocessing import normalize
from extl.models.util import regularize_matrix, is_pos_def

def baseline_experiments_synthetic():
  results = dict()
  data = load_data_set('synthetic')

  # base line
  clf = get_classifier('logistic')
  clf.fit(data.XS, data.yS)
  yP = clf.predict(data.XT)
  score = accuracy_score(yP, data.yT)
  results['logistic'] = score

  # transfer learning models.
  classifiers = ['tca', 'suba', 'rba', 'flda', 'tcpr']
  for classifier in classifiers:
    clf = get_classifier(classifier)
    clf.fit(data.XS, data.yS, data.XT)
    yP = clf.predict(data.XT)
    score = accuracy_score(yP, data.yT)
    results[classifier] = score


  return results


def baseline_experiments(data_set, source=None, target=None):

  data = load_data_set(name=data_set, source=source, target=target)

  XS = data.XS
  XT = data.XT

  m = XS.shape[0]

  X = np.vstack([XS, XT])
  X = X / np.max(X)

  XS = X[:m,:]
  XT = X[m:,:]


  random_state = 0

  nTL = 200

  pos_inds = np.where(data.yT > 0)[0]
  neg_inds = np.where(data.yT < 0)[0]

  np.random.seed(random_state)
  np.random.shuffle(pos_inds)

  np.random.seed(random_state)
  np.random.shuffle(neg_inds)

  pos_inds_l = pos_inds[:nTL]
  neg_inds_l = neg_inds[:nTL]

  pos_inds_u = pos_inds[nTL:]
  neg_inds_u = neg_inds[nTL:]

  inds_l = np.hstack([pos_inds_l, neg_inds_l])
  inds_u = np.hstack([pos_inds_u, neg_inds_u])


  _XS = np.vstack([XS, XT[inds_l,:]])
  _XT = XT[inds_u,:]

  _yS = np.hstack([data.yS , data.yT[inds_l]])
  _yT = data.yT[inds_u]



  print(_XS.shape, _XT.shape, _yS.shape, _yT.shape)



  # XS = XS/np.max(XS)
  # XT = XT/np.max(XT)




  # print(XS.sum(axis=1),XS.sum(axis=1).shape)
  # print(XS.sum(axis=0), XS.sum(axis=0).shape)






  # X_transform = TruncatedSVD(n_components=1000).fit_transform(X)


  yS = _yS
  yT = _yT
  XS = _XS
  XT = _XT


  #-------------------------------------------
  results = []

  clf = get_classifier('logistic')
  clf.fit(XS, yS)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, yT))

  clf = get_classifier('svm')
  clf.fit(XS, yS)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, yT))

  from extl.models.suba import SubspaceAlignedClassifier
  clf = SubspaceAlignedClassifier(num_components=1500, loss='logistic', l2=10)
  clf.fit(XS, yS, XT)
  yp = clf.predict(XT)
  results.append(accuracy_score(yp, yT))

  from extl.models.iw import ImportanceWeightedClassifier
  iwe = ['lr', 'nn', 'kmm']
  for _iwe in iwe:
    clf = ImportanceWeightedClassifier(iwe=_iwe, loss='logistic')
    clf.fit(XS, yS, XT)
    yp = clf.predict(XT)
    results.append(accuracy_score(yp, yT))

  for _iwe in iwe:
    clf = ImportanceWeightedClassifier(iwe=_iwe, loss='hinge')
    clf.fit(XS, yS, XT)
    yp = clf.predict(XT)
    results.append(accuracy_score(yp, yT))



  print(results)
  return results



