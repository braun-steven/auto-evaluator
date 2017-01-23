#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This module contains the evaluation classifier.
It implements a machine learning based approach to learn pattern and recognize
whether an article should be evaluated as 'wrong' or  not.


Usage:
    classifier_eval -h
    classifier_eval -c <config>

Options:
    -h, --help                                  Show this screen.
    -c <config>, --config <config>              Use configfile <config>.
(c) 2016 steven-lang@gmx.de
"""
import json
import logging
import pickle
from time import time

import numpy as np

from my_ensembles import EOCClassifier
from data_helper import get_data_from_elasticsearch
from data_helper import parse_hits
from classifier_rest import setup_app
from combiner import combine_results
from docopt import docopt
import elasticsearch
from version import __version__
import config_helper
import sys
from typing import Any, Union, Iterable, Tuple, SupportsFloat


log = logging.getLogger(__name__)

class EvalClassifier(object):
    def __init__(self, config):
        # type: (Dict[str, Any]) -> None
        """
        Initializes this classifier with the config
        :param config: Classifier configuration
        """
        self.name = 'EvalClassifier'
        self.version = __version__
        self.__config = config
        self.__clf = EOCClassifier(n_jobs=config.get('n_jobs'),
                                   aggregate='weighted',
                                   tune_parameters=config.get(
                                       'tune_params'))
        self.__threshold = float(config.get('threshold'))
        self.__es = elasticsearch.Elasticsearch(host=config.get('host'),
                                                port=config.get('port'))

        self.__setup()

        pass

    def __setup(self):
        # type: () -> None
        """
        Initial setup
        Retrain the model or load it from a serialized object
        :return:
        """
        print(self.__config)
        if self.__config.get('retrain') == 1:
            self.__build()
        else:
            self.__load_classifier(config.get('model_path'))

    def __build(self):
        # type: () -> None
        """
        Starts the training process of the underlying algorithm
        :return:
        """
        log.info('Started building the classifier on the training data.')

        log.debug('Getting the data from elasticsearch.')
        X, y, _ = get_data_from_elasticsearch(
            es=self.__es,
            index=self.__config.get('train_index')
        )

        np.random.seed(42)
        np.random.shuffle(X)
        np.random.seed(42)
        np.random.shuffle(y)

        X = X[10000:]
        y = y[10000:]

        log.debug('Training size: {}'.format(X.shape[0]))

        log.debug('Fitting the classifier to the data.')
        self.__clf.fit(X, y)

        log.debug('Saving the classifier.')
        self.__save_classifier(str(self.__config.get('model_path')))

    def classify(self, docs):
        # type: (Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]
        """
        Predicts the class "evaluated" of the given documents
        :param docs: Documents containing the hits from an elastic search query
        :return: List of 'wrong's or 'right/invalid's for the input documents
        """
        log.info('Started classifying documents.')
        log.debug('Parsing documents.')

        sh_config = json.load(open(str(self.__config.get(
            'streamhandler_config_path')), 'r'))  # type: Dict[str, Any]

        # Combine the documents
        combined_docs = combine_results(docs, sh_config)
        X, _, _ = parse_hits(docs=combined_docs,
                             training=False,
                             sampling=100)

        log.debug('Predicting the class probabilities.')
        probas = self.__clf.predict_proba(
            X)  # type: Iterable[Tuple[float, float]]

        results = []  # type: List[Dict[str, Any]]
        for doc, p in zip(combined_docs, probas):
            result = dict(hits=[])  # type: Dict[str, Any]
            if p[1] > self.__threshold:
                label = 'right/invalid'
                confidence = p[1]
            else:
                label = 'wrong'
                confidence = p[0]

            result['hits'].append(
                dict(
                    confidence=confidence,
                    classified_date=time(),
                    classifier_version=self.version,
                    model_version="nyi",
                    evaluated=label,
                )
            )
            results.append(dict(_id=doc['_id'], result=result))
        return results

    def __save_classifier(self, path):
        # type: (str) -> None
        """
        Stores the classifier at a given path
        :param path: Path for the serialization of the classifier
        :return:
        """
        log.debug('Saving the classifier at {}'.format(path))
        pickle.dump(self.__clf, open(path, 'wb'))

    def __load_classifier(self, path):
        # type: (str) -> None
        """
        Loads the classifier from a given path
        :param path: Path of the serialized classifier
        :return:
        """
        log.debug('Loading the classifier from {}'.format(path))
        self.__clf = pickle.load(open(path, 'rb'))




if __name__ == '__main__':

    # Setup config
    args = docopt(__doc__, version=__name__ + ' ' + __version__)
    config = config_helper.read_config(fname=args.get('--config'))
    config_helper.configure_logging(config=config)

    # Setup classifier
    classifier = EvalClassifier(config=config)

    # Setup app
    app = setup_app(classifier)
    port = config.get('rest_port')
    if 'debug' in sys.argv:
        app.run(debug=True, port=port)
    else:
        print("WARNING: running on public interface!")
        app.run(host='0.0.0.0', port=port)
