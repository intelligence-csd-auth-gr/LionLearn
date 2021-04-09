import math
import numpy as np
import CHIRPS.datasets as ds
import CHIRPS.routines as rt
import CHIRPS.structures as strcts
from CHIRPS import p_count_corrected
from CHIRPS import config as cfg

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from xgboost.sklearn import XGBClassifier # not compatible with latest scikit-learn versions, older versions break CHIRPS code at input matrix types
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support, accuracy_score

from CHIRPS import if_nexists_make_dir, if_nexists_make_file, chisq_indep_test, p_count_corrected
from CHIRPS.plotting import plot_confusion_matrix

from CHIRPS import config as cfg

# bug in sk-learn. Should be fixed in August
# import warnings
# warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
def calculate_weights(arr):
    def weight(err):
        err_value = (1-err)/err
        return(0.5 * math.log(err_value))

    vweight = np.vectorize(weight)

    return(vweight(arr))

# def _samme_proba(estimator, n_classes, X):
#     """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].
#     References
#     ----------
#     .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
#     """
#     proba = estimator.predict_proba(X)
#
#     # Displace zero probabilities so the log is defined.
#     # Also fix negative elements which may occur with
#     # negative sample weights.
#     proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
#     log_proba = np.log(proba)
#
#     return (n_classes - 1) * (log_proba - (1. / n_classes)
#                               * log_proba.sum(axis=1)[:, np.newaxis])
#
#
#
#
#             if self.algorithm == 'SAMME.R':
#             # The weights are all 1. for SAMME.R
#             pred = sum(_samme_proba(estimator, n_classes, X)
#                        for estimator in self.estimators_)
#         else:   # self.algorithm == "SAMME"
#             pred = sum((estimator.predict(X) == classes).T * w
#                        for estimator, w in zip(self.estimators_,
#                                                self.estimator_weights_))
#
#         pred /= self.estimator_weights_.sum()
#         if n_classes == 2:
#             pred[:, 0] *= -1
#             return pred.sum(axis=1)
#         return pred

# SAMME.R get weights

# SAMME - use existing weights

# weights and taking into account the discriminative power of the base learner
# in SAMME.R thiis is directly related
