from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.metrics import average_precision_score
from src.utils import pre_rec_f1, get_best_threshold
import numpy as np
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import make_scorer


def train_svc(train_data, valid_data, train_label, valid_label, kernel='rbf'):
    params = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': ['scale']}
    svc = SVC(kernel=kernel, max_iter=10000)
    clf = GridSearchCV(svc, params, scoring='average_precision')

    clf.fit(train_data, train_label)

    valid_score = clf.decision_function(valid_data)
    th, _ = get_best_threshold(valid_label, valid_score)

    return clf.best_estimator_, th


def test_svc(test_data, test_label, clf, th):
    pre_score = clf.decision_function(test_data)
    ap = round(average_precision_score(test_label, pre_score), 4)

    pre_label = np.zeros(test_label.shape, dtype=int) - 1
    idx = np.where(pre_score >= th)[0]
    pre_label[idx] = 1
    pre, rec, f1 = pre_rec_f1(test_label, pre_label)
    return [ap, pre, rec, f1]


def reg_sper(y_true, y_pred):
    sper = spearmanr(y_true, y_pred)[0]
    return sper


def reg_ken(y_true, y_pred):
    ken = kendalltau(y_true, y_pred)[0]
    return ken

from sklearn import preprocessing

def train_svr(train_data, train_label, kernel='rbf'):
    params = {'C': [0.01, 0.1, 1, 5, 10, 50, 100, 1000], 'gamma': ['scale', 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    svr = SVR(kernel=kernel, degree=1, max_iter=10000)
    clf = GridSearchCV(svr, params, scoring=make_scorer(reg_sper), cv=3)
    clf.fit(train_data, train_label)
    return clf.best_estimator_


def test_svr(test_data, test_label, clf):
    pre_score = clf.predict(test_data)
    sper = round(spearmanr(test_label, pre_score)[0], 4)
    ken = round(kendalltau(test_label, pre_score)[0], 4)
    return [sper, ken]
