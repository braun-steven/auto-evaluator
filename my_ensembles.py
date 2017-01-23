import logging
from time import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from typing import Any, Union, Iterable, Tuple, SupportsFloat

log = logging.getLogger(__name__)


class EOCClassifier(BaseEstimator, ClassifierMixin):
    """Votes over classifier confidences between naive bayes, random
    forest, SGD, logistic regression and SVM with an rbf kernel."""

    def __init__(self, tune_parameters=False, n_jobs=1, aggregate='weighted'):
        # type: (bool, int, str) -> None
        """
        Init the classifier.
        :param tune_parameters: Wether to use gridsearch for parameter
        tuning. This may take exponentially longer than without.
        :param n_jobs: Number of parallel jobs if possible.
        :param aggregate: How to aggregate over the probabilities of the
        single classifiers.
        """
        self.aggregate = aggregate
        self.n_jobs = n_jobs

        if tune_parameters:
            self._init_tuned_clfs()
        else:
            self._init_clfs()

        pass

    def _init_clfs(self):
        # type: () -> None
        """
        Initializes the classifiers
        :return:
        """
        nb = MultinomialNB(fit_prior=True,
                           alpha=0.78)

        rf = RandomForestClassifier(n_estimators=200,
                                    random_state=42,
                                    criterion='gini',
                                    n_jobs=self.n_jobs)

        logreg = LogisticRegression(C=2,
                                    intercept_scaling=0.22,
                                    solver='liblinear',
                                    n_jobs=self.n_jobs)

        svm = SVC(C=8,
                  kernel='rbf',
                  gamma=0.03125,
                  probability=True)

        self.clfs = [
            nb,
            rf,
            logreg,
            svm
        ]

    def _init_tuned_clfs(self):
        # type: () -> None
        """
        Initialize tuned classifier with GridSearchCV
        This can take exponentially longer in the training phase than
        self.init_clfs().
        This should be done after the training data has changed a lot and
        maybe new pattern can be recognized, the data is more noisy/erroneous
        :return:
        """

        scorer = 'accuracy'
        nb = GridSearchCV(
            estimator=MultinomialNB(fit_prior=True),
            n_jobs=self.n_jobs,
            param_grid={
                'alpha': [x / float(100) for x in range(0, 101, 10)]
            },
            scoring=scorer
        )

        rf = GridSearchCV(
            estimator=RandomForestClassifier(n_estimators=200,
                                             random_state=42),
            n_jobs=self.n_jobs,
            param_grid={
                'criterion': ('gini', 'entropy')
            },
            scoring=scorer
        )

        logreg = GridSearchCV(
            estimator=LogisticRegression(solver='liblinear'),
            n_jobs=self.n_jobs,
            param_grid={
                'intercept_scaling': [x / float(100) for x in range(1, 101, 5)],
                'C': [2 ** x for x in range(-3, 19, 2)]
            },
            scoring=scorer
        )

        svm = GridSearchCV(
            estimator=SVC(probability=True),
            param_grid={'C': [2 ** x for x in range(-3, 17, 2)],
                        'gamma': [2 ** x for x in range(-13, 3, 2)],
                        'kernel': ['rbf']},
            n_jobs=self.n_jobs,
            scoring=scorer
        )

        self.clfs = [
            nb,
            rf,
            logreg,
            svm
        ]

    def set_params(self, **params):
        return super(EOCClassifier, self).set_params(**params)

    def get_params(self, deep=True):
        return super(EOCClassifier, self).get_params(deep)

    def fit(self, X, y):
        # type: (np.ndarray[Union[int, float]], List[int]) -> EOCClassifier
        """
        For the classifier on the training data
        :param X: Training instances
        :param y: Training labels
        :return: self
        """
        for clf in self.clfs:
            t0 = time()
            log.debug('Fitting {}'.format(clf.__class__.__name__))
            clf.fit(X, y)
            log.debug('Fitting {} took {}s'.format(clf.__class__.__name__,
                                               time() - t0))

        # Get GridSearchCV best_estimators
        best_clfs = []
        for clf in self.clfs:
            if isinstance(clf, GridSearchCV):
                best_clfs.append(clf.best_estimator_)
                print('Found best params for {}'.format(
                    clf.best_estimator_.__class__.__name__))
                print(clf.best_params_)
            else:
                best_clfs.append(clf)

        self.clfs = best_clfs
        return self

    def _aggregrate_probabilites(selfs, tuples, method='weighted'):
        # type: (np.ndarray[float], str) -> List[float]
        """
        Decides how to aggregate over the probability predictions from the
        different classifiers
        :param tuples: List of probability distributions for the classes
        given by each of the classifier
        :param method: Can be 'max' or 'weighted'
        :return: Aggregated probability distribution
        """

        best_tup = [0.0, 0.0]

        if method == 'max':
            # Find the highest probability
            for tup in tuples:
                if max(tup) > max(best_tup):
                    best_tup = tup
        elif method == 'weighted':
            # Weight over all probabilities
            for i in range(tuples.shape[1]):
                best_tup[i] = np.sum(tuples[:, i]) / float(len(tuples))

        return best_tup

    def predict(self, X):
        # type: (np.ndarray[Union[int, float]]) -> np.ndarray[int]
        """
        Predicts the labels for all instances in X
        :param X: List of input instances
        :return: List of labels, shape: (len(X), 1)
        """
        probas = self.predict_proba(X)
        res = []
        for p0, p1 in probas:
            if p0 > p1:
                res.append(0)
            else:
                res.append(1)

        return np.array(res)

    def predict_proba(self, X):
        # type: (np.ndarray[Union[int, float]]) -> np.ndarray[List[float]]
        """
        Predicts the probabilities for all classes of all instances in X
        :param X: List of input instances
        :return: List of probabilities, shape: (len(X), num_classes)
        """
        # probas = []
        # for clf in self.clfs:
        #     probas.append(clf.predict_proba(X))

        probas = np.array([clf.predict_proba(X) for clf in self.clfs])

        res = []

        for i in range(0, probas.shape[1]):
            res.append(self._aggregrate_probabilites(tuples=probas[:, i],
                                                     method=self.aggregate))

        return np.array(res)
