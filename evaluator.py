import pickle

import numpy as np
import matplotlib

# Force matplotlib to not use any Xwindows backend. call before any other
# plot import
from sklearn import cross_validation
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import data_helper

matplotlib.use('Agg')
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp

# File prefix
name = ''

# Outfile
f = None


def set_outfile(out):
    """
    Sets the logfile
    :param out: logfile descriptor
    :return:
    """
    global f
    f = out


def pr(s):
    """
    Prints some stuff to the stdout and a file 'f'
    :param s:
    :return:
    """
    print(s)
    f.write(str(s) + '\n')


def set_name(n):
    """
    Sets the file output name
    :param n:
    :return:
    """
    global name
    name = n


def print_metrics(y_true, y_pred, y_pred_proba):
    """
    Prints some metrics about the evaluation
    :param y_true: true labels
    :param y_pred: predicted labels
    :param y_pred_proba: prediction probabilities
    :return:
    """
    target_names = ['wrong', 'right']
    out = ''
    out += 'Classification report: \n'
    out += classification_report(y_true, y_pred, target_names=target_names)
    out += '\n'
    out += 'Accuracy score: {}'.format(accuracy_score(y_true, y_pred))
    out += '\n'
    out += 'Log loss: {}'.format(log_loss(y_true, y_pred_proba))
    pr(out)


def plot_cm_curve(y_test, y_pred):
    cond_pos = float(len([x for x in y_test if x == 1]))
    cond_neg = float(len([x for x in y_test if x == 0]))
    fpr, tpr, ths = roc_curve(y_test, y_pred[:, 1])

    data = []
    for th in [x/float(100) for x in range(0, 101)]:
        # Skip erroneous thresholds
        if th > 1:
            continue

        y_p = []
        for p0, p1 in y_pred:
            y_p.append(1 if p1 > th else 0)

        cm = confusion_matrix(y_test, y_p)
        tn = cm[0, 0] / cond_neg
        fp = cm[0, 1] / cond_neg
        fn = cm[1, 0] / cond_pos
        tp = cm[1, 1] / cond_pos

        data.append([tn, fp, fn, tp, th])

    data = np.array(data)

    ths = data[:, 4]
    plt.clf()
    plt.figure()
    plt.plot(ths, data[:, 0], label='TN/N')
    plt.plot(ths, data[:, 1], label='FP/N')
    plt.plot(ths, data[:, 2], label='FN/P')
    plt.plot(ths, data[:, 3], label='TP/P')
    #plt.plot(ths, data[:, 0] - data[:, 2], label='TN/N : FN/P')
    plt.title('CM over all thresholds')
    plt.xlabel('Thresholds')
    plt.ylabel('Percentages')
    plt.legend(loc="lower right")
    plt.savefig(name + 'cm_ths.png')
    plt.clf()


def add_plot_roc(y_test, y_pred):
    """
    Adds the roc curve to the plot
    :param y_test: true labels
    :param y_pred: prediction probabilities
    :return:
    """

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    target_names = ['wrong', 'right']
    # Plot all ROC curves
    plt.subplot(221)
    for i in [0, 1]:
        # Compute ROC curve and ROC area for each class
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i],
                 label='{} (AUC:{:0.2f})'.format(target_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        'ROC Curve')
    plt.legend(loc="lower right")


def plot_feature_importance():
    """
    Plots the feature importance of the classifier
    :return:
    """
    plt.clf()
    forest = pickle.load(open(name + '/clf.p', 'rb'))

    if not hasattr(forest, 'feature_importances_'):
        print('No features importances found. Skipping plot for '
              'feature importances ...')
        return

    importances = forest.feature_importances_

    n = len(importances)
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    pr("Feature ranking:")

    for f in range(n):
        pr(
            "%d. feature %d (%f)" % (
                f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(n), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(n), indices)
    plt.xlim([-1, n])
    plt.savefig(name + 'feat_imp.png')


def add_plot_confusion_matrix(y_test, y_pred):
    """
    Adds the confusion matrix and the normalized confusion matrix to the plot
    :param y_test: true labels
    :param y_pred: predicted labels
    :return:
    """

    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        names = ['wrong', 'right']
        tick_marks = np.arange(len(names))
        plt.xticks(tick_marks, names, rotation=45)
        plt.yticks(tick_marks, names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    pr('Confusion matrix, without normalization')
    pr(cm)
    plt.subplot(223)
    plot_confusion_matrix(cm)

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    pr('Normalized confusion matrix')
    pr(cm_normalized)
    plt.subplot(224)
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')


def add_plot_precision_recall_curve(y_test, y_pred):
    """
    Adds the prc to the plot
    :param y_test: true labels
    :param y_pred: prediction probabilities
    :return:
    """

    precision = dict()
    recall = dict()
    average_precision = dict()

    # Plot Precision-Recall curve for each class
    plt.subplot(222)

    target_names = ['wrong', 'right']

    for i in [0, 1]:
        # Compute Precision-Recall and plot curve
        precision[i], recall[i], _ = precision_recall_curve(y_test,
                                                            y_pred[:, i],
                                                            pos_label=i)
        average_precision[i] = average_precision_score(
            [(1 - y) if i == 0 else y for y in y_test],
            y_pred[:, i]
        )
        plt.plot(recall[i], precision[i],
                 label='{} (AUC:{:0.2f})'.format(target_names[i],
                                                 average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")


def plot_calibration_curve(est, nam, s=100):
    plt.clf()
    X, y = data_helper.get_data('data-proposal.json', s)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.33, random_state=42)

    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv='prefit', method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv='prefit', method='sigmoid')


    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, nam in [(est, nam),
                     (isotonic, nam + ' + Isotonic'),
                     (sigmoid, nam + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=1)
        print("%s:" % nam)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (nam, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=nam,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.savefig(name + 'prob_calib.png')
    plt.clf()


def evaluate(y_test, y_pred, y_pred_proba, s, clf, clf_type):
    """
    Starts the evaluation process
    :param y_test: true labels
    :param y_pred: predicted labels
    :param y_pred_proba: prediction probabilities
    :return:
    """

    plt.figure(figsize=(16, 12))

    # Print metrics
    print_metrics(y_test, y_pred, y_pred_proba)

    # Plot roc curve
    add_plot_roc(y_test, y_pred_proba)

    # Plot confusion matrix
    add_plot_confusion_matrix(y_test, y_pred)

    # Plot precision recall curve
    add_plot_precision_recall_curve(y_test, y_pred_proba)

    plt.savefig(name + 'eval.png')

    #plot_feature_importance()

    plot_cm_curve(y_test, y_pred_proba)

    #plot_calibration_curve(est=clf, nam=clf_type, s=s)
