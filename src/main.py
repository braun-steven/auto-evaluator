import datetime
import os

import matplotlib

# Force matplotlib to not use any Xwindows backend. call before any other
# plot import
import re
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from my_ensembles import EOCClassifier

matplotlib.use('Agg')

from sklearn import  model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import argparse
from time import gmtime, strftime, time
import numpy as np
import evaluator
import data_helper
import pickle

# File prefix
name = ''

# Output file
f = None


def pr(s):
    """
    Prints some stuff to the stdout and a file 'f'
    :param s:
    :return:
    """
    print(s)
    f.write(str(s) + '\n')
    f.flush()


def build_and_eval_clf(path, clf_type, samples):
    """
    Starts the building process of the classifier
    :param path: data input path
    :param clf_type: classifier type
    :param samples: percentage of the data
    :return:
    """
    print('Getting the data...')
    X, y = data_helper.get_data(path, samples)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.33, random_state=42)

    print('Training size: {}, Test size: {}'.format(X_train.shape[0],
                                                    X_test.shape[0]))

    print('Building the classifier...')
    clf = get_clf(clf_type)
    if isinstance(clf, GridSearchCV):
        clf = clf.fit(X_train, y_train)
        pr(clf_type + ' BEST PARAMS: ')
        pr(clf.best_params_)
        clf = clf.best_estimator_
        pr('Best estimator:')
        pr(clf)
    else:
        clf = clf.fit(X_train, y_train)

    print('Serializing the classifier...')
    # Serialize clf
    pickle.dump(clf, open(name + 'clf.p', 'wb'))

    print('Making predictions...')
    y_pred = clf.predict(X_test)

    if hasattr(clf, 'predict_proba'):
        y_pred_proba = clf.predict_proba(X_test)
    else:
        y_pred_proba = [(1, 0) if x == 1 else (0, 1) for x in y_pred]

    print('Starting evaluation...')
    # Use unseen parameter for evaluation
    evaluator.evaluate(y_test=y_test,
                       y_pred=y_pred,
                       y_pred_proba=y_pred_proba,
                       clf=clf,
                       clf_type=clf_type,
                       s=samples)


def auc_score(y_test, y_pred):
    return roc_auc_score(y_test, y_pred)


def get_clf(type):
    """
    Get clf by type
    :param type: clf type
    :return: clf
    """


    scorer = make_scorer(auc_score)

    default = MultinomialNB(fit_prior=True)
    clf = {
        # Gridsearch over C, GAMMA and kernel
        'svm': GridSearchCV(SVC(probability=True),
                            param_grid=[
                                # {'C': [2 ** x for x in range(-3, 19, 2)],
                                #  'kernel': ['linear']},
                                {'C': [2 ** x for x in range(-3, 19, 2)],
                                 'gamma': [2 ** x for x in range(-15, 3, 2)],
                                 'kernel': ['rbf']},
                            ],
                            n_jobs=n_jobs, verbose=3, cv=2),

        'svmrbf': SVC(C=8, kernel='rbf', gamma=0.03125, probability=True),

        'nb': MultinomialNB(fit_prior=True, alpha=0.78),

        'rf': RandomForestClassifier(n_estimators=200, random_state=42,
                                     criterion='gini', n_jobs=n_jobs),

        'eoc': EOCClassifier(aggregate='weighted', n_jobs=n_jobs),

        'eocgs': EOCClassifier(aggregate='weighted', n_jobs=n_jobs,
                               tune_parameters=True),

        'sgd': GridSearchCV(estimator=SGDClassifier(shuffle=True,
                                                    loss='modified_huber',
                                                    learning_rate='optimal',
                                                    n_iter=10000),
                            param_grid={
                                'penalty': ('l2', 'elasticnet'),
                                'average': [True, False],
                                'alpha': 10.0 ** -np.arange(2, 7),
                            }, n_jobs=n_jobs, verbose=3),

        'sgd1': SGDClassifier(loss='modified_huber', penalty='l2',
                              n_iter=10000, shuffle=True, verbose=3, n_jobs=4,
                              random_state=42),

        'logreg': GridSearchCV(
            estimator=LogisticRegression(solver='sag'), param_grid=[

                {
                    'solver': ['liblinear'],
                    'intercept_scaling': [x / float(100) for x in range(1, 101,
                                                                        3)],
                    'C': [2]
                }], n_jobs=4, verbose=3)
    }.get(type, default)
    pr('Chosen classifier: ')
    pr(clf)
    return clf


n_jobs = 4


def main(path, clf_type, tag, samples, args):
    """
    Main method
    :param path: path to the input data
    :param clf_type: type of the classifier
    :param tag: tag which should be added to the folder name
    :param samples: percentage of samples
    :param args: other arguments from the cmd line
    :return:
    """

    t0 = time()

    t = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    global name
    name = 'vis/' + t + '-' + clf_type + '_' + str(samples) + '_' + tag + '/'
    evaluator.set_name(name)

    data_helper.ensure_dir('vis/')
    data_helper.ensure_dir(name)

    global f
    f = open(name + 'out.log', 'w')
    evaluator.set_outfile(f)

    global n_jobs
    n_jobs = int(args.n_jobs)
    if not n_jobs:
        n_jobs = 4

    pr('Arguments: {}'.format(args))
    build_and_eval_clf(path, clf_type, samples)
    f.close()

    print("done in %0.3fs" % (time() - t0))


def filter_surnames():
    f = open('/home/slang03/Downloads/names.html', 'r')
    lines = []
    for line in f.readlines():
        lines.append(line)

    lines = filter(lambda l: 'title=' in l, lines)
    lines = filter(lambda l: re.match('<li><a href=.*">.*</a></li>', l) is not
                             None, lines)
    lines = map(lambda l: re.sub('<li><a href=.*">', '', l), lines)
    lines = map(lambda l: re.sub('</a>.*\n', '', l), lines)
    out = open('names.txt', 'w')
    out.write('\n'.join(lines))
    f.close()
    out.close()


if __name__ == '__main__':

    # Set description
    parser = argparse.ArgumentParser(description='Apply machine learning on '
                                                 'article evaluation')

    # Set arguments
    parser.add_argument('-d', help='Path to the data')
    parser.add_argument('-clf', help='Classifiertype (May be "gb", "svm", '
                                     '"nb" or "rf")')
    parser.add_argument('-t', help='Specifies a tag for the output')
    parser.add_argument('-clean', help='Deletes the cached dataset')
    parser.add_argument('-s', help='Sampling rate in percent')
    parser.add_argument('-n_jobs', help='Number of parallel jobs if available')
    args = parser.parse_args()
    data_path = args.d
    tag = args.t
    if not tag:
        tag = ''

    n_jobs = args.n_jobs or 1

    if not n_jobs:
        n_jobs = 4

    # Remove cached file
    if args.clean:
        data_helper.delete_cache(args.clean)

    main(data_path, args.clf, tag, float(args.s), args)
