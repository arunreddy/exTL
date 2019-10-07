from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

from extl.models.da.iw import ImportanceWeightedClassifier
from extl.models.da.tca import TransferComponentClassifier
from extl.models.da.suba import SubspaceAlignedClassifier
from extl.models.da.scl import StructuralCorrespondenceClassifier
from extl.models.da.rba import RobustBiasAwareClassifier
from extl.models.da.flda import FeatureLevelDomainAdaptiveClassifier
from extl.models.da.tcpr import TargetContrastivePessimisticClassifier
import logging

logger = logging.getLogger(__name__)

def get_classifier(classifier):

  logger.info('Fetching the classifer {}'.format(classifier))

  # Select adaptive classifier
  if classifier == 'iw':
    # Call an importance-weighted classifier
    clf = ImportanceWeightedClassifier(iwe='lr', loss='logistic')

  elif classifier == 'tca':
    # Classifier based on transfer component analysis
    clf = TransferComponentClassifier(loss='logistic', mu=1.)

  elif classifier == 'suba':
    # Classifier based on subspace alignment
    clf = SubspaceAlignedClassifier(loss='hinge',l2=1.0, num_components=500)

  elif classifier == 'scl':
    # Classifier based on subspace alignment
    clf = StructuralCorrespondenceClassifier(loss='hinge', num_pivots=500, num_components=50)

  elif classifier == 'rba':
    # Robust bias-aware classifier
    clf = RobustBiasAwareClassifier(l2=0.1, max_iter=1000)

  elif classifier == 'flda':
    # Feature-level domain-adaptive classifier
    clf = FeatureLevelDomainAdaptiveClassifier(l2=0.1, max_iter=1000)

  elif classifier == 'tcpr':
    # Target Contrastive Pessimistic Classifier
    clf = TargetContrastivePessimisticClassifier(l2=0.1)

  elif classifier == 'logistic':
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)

  elif classifier == 'svm':
    clf = LinearSVC()


  elif classifier == 'svm-rbf':
    clf = SVC(kernel='rbf')

  else:
    raise ValueError('Classifier not recognized.')


  return clf
