import pickle

import scipy
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.datasets import svmlight_format, dump_svmlight_file
import matplotlib
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import MultinomialNB

import evaluator
from my_ensembles import EOCClassifier

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pprint
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from sklearn.linear_model import SGDClassifier
import data_helper
from data_helper import get_data


def plot_tfidf_vect_scorings():
    with open('/home/slang03/cv_tfid_rf_full_df08.txt', 'r') as f:
        lines = []
        for line in f.readlines():
            line = line.replace('\n', '')
            splits = line.split(', ')
            splits = [x.split('=')[1] for x in splits]

            ngram = int(splits[0][3])
            max_feats = int(splits[1])
            score = float(splits[3].split(' -')[0])
            lines.append([ngram, max_feats, score])

        lines = np.array(lines)

        plt.figure()
        unigrams = np.array([x for x in lines if x[0] == 1])
        bigrams = np.array([x for x in lines if x[0] == 2])
        plt.plot(unigrams[:, 1], unigrams[:, 2], label='unigrams')
        plt.plot(bigrams[:, 1], bigrams[:, 2], label='bigrams')
        plt.title('Unigrams vs bigrams with different number of features')
        plt.xlabel('features')
        plt.ylabel('accuracy')
        plt.legend(loc="lower right")
        plt.savefig('unigram_vs_bigram_rf_df08.png')


def plot_num_instances_scorings():
    data = []
    for s in range(5, 101, 5):
        data_helper.delete_cache()
        X, y = get_data('data-proposal.json', sampling=s)
        clf = SGDClassifier()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=0.33, random_state=42)

        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        zo_loss = zero_one_loss(y_test, y_pred)

        vals = [s, acc, zo_loss]
        pprint.pprint(vals)
        data.append(vals)

    data = np.array(data)
    plt.clf()
    ss = np.array(map(lambda t: t[0], data))
    accs = np.array(map(lambda t: t[1], data))
    zos = np.array(map(lambda t: t[2], data))

    plt.plot(ss, accs, label='Accuracy')
    plt.plot(ss, zos, label='Zero-One-Loss')
    plt.title('Accuracy vs Zero-One-Loss vs num instances')
    plt.ylabel('Metrics')
    plt.xlabel('Num instances')
    plt.legend(loc="lower right")
    plt.savefig('num_inst.png')


def get_false_negs():
    plt.figure(figsize=(16, 12))
    print('Loading data')
    X, y, content = data_helper.get_data_with_content('data-proposal.json', 100)

    print('Splitting in test/train set')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.33, random_state=42)

    print('Loading clf')
    clf = pickle.load(open('vis/2016-10-10_06-43-58-eoc_100.0_full_eoc_for_testing/clf.p', 'rb'))

    clfs_copy = clf.clfs

    for classifier in clfs_copy + ['nothing']:
        if classifier != 'nothing':
            clf.clfs.remove(classifier)

        print('Curently without Classifier: {}'.format(
            classifier.__class__.__name__))
        print('Predicting {} test instances'.format(X_test.shape[0]))
        y_pred_proba = clf.predict_proba(X_test.toarray())
        clf.clfs.append(classifier)
        print('Finished')

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        label = "without {}".format(classifier.__class__.__name__)
        plt.plot(fpr, tpr,
                 label='{} (AUC:{:0.2f})'.format(label, roc_auc))
    plt.title('TN vs FN')
    plt.xlabel('False Negative Rate')
    plt.ylabel('True Negative Rate')
    plt.legend(loc="lower right")
    plt.savefig('ROC-EOC.png')
    return

    content_train, content_test, _, _ = \
        cross_validation.train_test_split(
            content, y, test_size=0.33, random_state=42)

    clf = EOCClassifier()

    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)

    true_neg_rates = []
    fn_invs_rates = []
    fn_rights_rates = []
    ths = []

    for i in range(0, 101):
        th = i / float(100)
        ths.append(th)
        a = lambda p: 1 if p[1] > th else 0
        mapped = map(a, y_pred_proba)
        zipped = zip(y_test, mapped, content_test)

        filt = lambda t: t[0] == 1 and t[1] == 0
        filtered = filter(filt, zipped)

        invs = [x for x in filtered if '>>EVALUATED>>: invalid' in x[2]]
        rights = [x for x in filtered if '>>EVALUATED>>: right' in x[2]]

        fn_invs = len(invs)
        fn_rights = len(rights)
        fns = len(filtered)

        tns = len(filter(lambda t: t[0] == 0 and t[1] == 0, zipped))
        negs = len(filter(lambda t: t[0] == 0, zipped))

        true_neg_rates.append(tns / float(negs))
        fn_invs_rates.append(fn_invs / float(10 ** 10 if fns == 0 else fns))
        fn_rights_rates.append(fn_rights / float(10 ** 10 if fns == 0 else fns))

    plt.figure()
    plt.plot(ths, true_neg_rates, label='tn_rate')
    plt.plot(ths, fn_invs_rates, label='fn_inv_rates')
    plt.plot(ths, fn_rights_rates, label='fn_right_rates')
    plt.title('')
    plt.ylabel('%')
    plt.xlabel('ths')
    plt.legend(loc="lower right")
    plt.savefig('fn_invs_vs_rights.png')

    # a = lambda p: 1 if p[1] > 0.25 else 0
    # mapped = map(a, y_pred_proba)
    # zipped = zip(y_test, mapped, content_test)
    #
    # filt = lambda t: t[0] == 1 and t[1] == 0
    # filtered = filter(filt, zipped)
    #
    # invs = [x for x in filtered if '>>EVALUATED>>: invalid' in x[2]]
    # rights = [x for x in filtered if '>>EVALUATED>>: right' in x[2]]
    # f = open('false_negs.txt', 'w')
    # for y_true, y_pred, content in filtered:
    #     s = '\n' + '>' * 80 + '\n'
    #     s += content.encode('UTF-8')
    #     s += '\n' + '<' * 80 + '\n'
    #     f.write(s)
    # f.close()
    return

def test_word2vec():
    data_helper.build_word2vec_model('data-proposal.json')


def main():
    # test_word2vec()
    get_false_negs()
    return


if __name__ == '__main__':
    main()
