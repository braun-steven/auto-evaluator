#!./env/bin/python2
# -*- coding: utf-8 -*-
import datetime
import logging
import os
import random

import progressbar
import re
from scipy.sparse import csr_matrix
import snowballstemmer
import json
import numpy as np
from urlparse import urlparse
import pickle
import elasticsearch

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import word2vec
from typing import Any, Union, Iterable, Tuple, SupportsFloat

# Classifier names of the current classifiers in the system
cls_names = ['query', 'websimilarity', 'exact_match', 'min_score',
             'spam_article']

# Split pattern to split at everything which is not a
split_pattern = re.compile('[^a-zA-Z0-9äöüß]')

# Load snowballstemmer german
stemmer = snowballstemmer.stemmer('german')

# sws = get_stemmed_stopwords()
log = logging.getLogger('classifier_eval')


def replace_umlauts(st):
    # type: (str) -> unicode
    """
    Replaces all umlauts in a given string
    :param st: string
    :return: string without umlauts (replaced)
    """
    # st = st.encode('utf-8')
    return st.lower().replace(u'ä', u'ae') \
        .replace(u'ö', u'oe') \
        .replace(u'ü', u'ue') \
        .replace(u'ß', u'ss')


def read(path):
    # type: (str) -> List[Dict[str, Any]]
    """
    Reads the output of an elasticsearch query
    :param path: data-path to the output
    :return: hits as generator
    """
    data = json.load(open(path, 'r'), encoding='utf-8')
    res = []
    for hit in data['hits']['hits']:
        res.append(hit)
    return res


def get_surnames():
    # type: () -> Iterable[str]
    """
    Reads a list of surnames from 'names.txt'
    :return: List of surenames
    """
    f = open('names.txt', 'r')
    return set(f.readlines())


def find_ngrams(company_name, n):
    # type: (str, int) -> List[str]
    """
    Produces a list of n-grams of the given string
    :param company_name: String which is to be splitted
    :param n: n
    :return: List of n-grams
    """
    return zip(*[company_name.split()[i:] for i in range(n)])


def get_content_cpyname_ngram_score(company_name, content, n):
    # type: (str, str, int) -> float
    """
    Matches the contentn with ngrams of the company name
    :param company_name: Company name
    :param content: Article content
    :param n: The 'n' of n-grams
    :return: Scoring
    """
    count = 0
    bigrams = find_ngrams(company_name, n)
    for bigram in bigrams:
        if ' '.join(bigram) in content:
            count += 1
    rel_count = 0 if bigrams == [] else count / float(len(bigrams))
    return rel_count


def build_word2vec_model(path, type, sampling=100):
    # type: (str, str, int) -> None
    """
    Builds a word2vec model
    :param path: load data from path
    :param type: type of content
    :param sampling: sampling rate
    :return:
    """
    r = open('{}_content.txt'.format(type), 'w')
    words = []

    # Set seed
    random.seed(42)
    for doc in read(path):

        # Apply sampling
        if random.randint(0, 100) > sampling:
            continue

        src = doc['_source']
        content = src['content'].lower()
        content = replace_umlauts(content)

        # Skip non right
        if type == 'right' and src['evaluated'] == 'wrong':
            continue
        if type == 'wrong' and (src['evaluated'] == 'right' or src[
            'evaluated'] == 'invalid'):
            continue

        for w in re.split(split_pattern, content):
            words.append(w)

    r.write(' '.join(words).encode('utf-8'))
    r.close()

    log.debug('Starting word2phrase ...')
    word2vec.word2phrase('{}_content.txt'.format(type),
                         '{}_content_phrases.txt'.format(type),
                         verbose=True)

    log.debug('Starting word2vec ...')
    word2vec.word2vec('{}_content_phrases.txt'.format(type),
                      '{}_w2v_model.bin'.format(type),
                      size=100, verbose=True, binary=1, min_count=1, cbow=0,
                      hs=1)


def parse_hits(docs, training=True, sampling=100):
    # type: (List[Dict[str, Any]], bool, int) -> Tuple[np.ndarray[np.ndarray[Any]],List[Any], List[Any]]
    """
    Parses a hit from the elasticsearch result
    :param doc: hit document
    :return: dictionary with relevant fields from the hit
    """

    # Data vector: shape = (n_samples, n_features)
    data = []

    # Target vector: shape = (n_samples, 1)
    target = []

    # Text corpus
    corpus = []
    #
    # right_w2v = word2vec.load('right_w2v_model.bin')
    # wrong_w2v = word2vec.load('wrong_w2v_model.bin')

    features = [
        'company_uid',
        'publisher',
        # 'query_clf_conf',
        # 'websimilarity_clf_conf',
        # 'exact_match_clf_conf',
        # 'min_score_clf_conf',
        # 'spam_article_clf_conf',
        'confidence_evaluated',
        'num_clfs',
        'title_companyname_match',
        'content_companyname_match',
        'num_snippets',
        # 'num_unique_words',
        # 'websim_found_name',
        'query_num_hl_terms',
        'websimilarity_num_hl_terms',
        'exact_match_num_hl_terms',
        'min_score_num_hl_terms',
        'spam_article_num_hl_terms',
        # 'query_websim_intersects',
        'company_content_bigrams',
        # 'company_content_trigrams',
        # 'weekday',
        # 'right_w2v_score',
        # 'wrong_w2v_score'
    ]
    # List of features which should be put into the
    # training data

    # Set seed
    random.seed(42)

    log.debug('Processing documents ...')

    bar = progressbar.ProgressBar(maxval=len(docs), \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    i = 0
    # Process each document
    for idx, doc in enumerate(docs):
        bar.update(idx)

        # Apply sampling
        if random.randint(0, 100) > sampling:
            continue

        # This is the dictionary in which all fields of an article is stored
        src = doc.get('_source', doc) # type: Dict[str, Any]

        # Prediction label
        evaluated = 0

        # Evaluated field should be in the training data
        if training:
            if src['evaluated'] == 'right':
                evaluated = 1
            elif src['evaluated'] == 'wrong':
                evaluated = 0
            elif src['evaluated'] == 'invalid':
                evaluated = 1
            else:
                log.warn(
                    'Warning: Training instance had no '
                    'field evaluated with "right" "wrong" or "invalid"'
                )
                continue

        company_uid = str(src['company_uid'])

        # Get publisher by domain
        parsed_uri = urlparse(src['source_id'])
        publisher = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

        # Get aggregate confidence
        confidence_evaluatd = sum(
            [float(el["confidence"]) for el in src["company_classifications"]])

        # Collect relevant source fields
        rel_src = {'evaluated': evaluated,
                   'company_uid': company_uid,
                   'publisher': publisher,
                   'confidence_evaluated': confidence_evaluatd} # type: Dict[str, Union[str, int, float]]

        # Init classifier results
        for cls_name in cls_names:
            rel_src[cls_name + '_clf_conf'] = 0.0

        # Add classifier results
        s = 'topic_classifications'
        if s in src:
            topic_clfs = src[s]
        else:
            topic_clfs = []

        if 'company_classifications' not in src:
            src['company_classifications'] = []

        rel_src['websimilarity_num_hl_terms'] = 0
        rel_src['num_clfs'] = len(src['classified_by'])

        # Edit the content for easier insight later on (logging)
        rel_src['content'] = '\n >>CPY-NAME>>: ' + src['company_name']
        rel_src['content'] += '\n >>TITLE>>: ' + src['title']
        rel_src['content'] += '\n >>EVALUATED>>: ' + src['evaluated']
        rel_src['content'] += '\n >>CONTENT>>: ' + src['content']
        content = src['content']

        # Match companyname-content bigrams
        rel_src['company_content_bigrams'] = get_content_cpyname_ngram_score(
            company_name=src['company_name'].lower(), content=content.lower(
            ), n=2)

        # Match companyname-content trigrams
        rel_src['company_content_trigrams'] = get_content_cpyname_ngram_score(
            company_name=src['company_name'].lower(), content=content.lower(

            ), n=3)

        rel_src['num_unique_words'] = len(set(content.split())) / float(len(
            content.split()))

        # Match title and company name and normalize it
        title = src['title'].lower()
        content = src['content'].lower()
        comp_name = src['company_name'].lower()
        company_name_terms = re.split(split_pattern, comp_name)
        comp_name_matches_title = 0.0
        comp_name_matches_content = 0.0
        for term in company_name_terms:
            if term in title:
                comp_name_matches_title += 1

            if term in content:
                comp_name_matches_content += 1
        num_name_terms = len(company_name_terms)
        comp_name_matches_title /= float(num_name_terms)
        comp_name_matches_content /= float(num_name_terms)
        rel_src['title_companyname_match'] = comp_name_matches_title
        rel_src['content_companyname_match'] = comp_name_matches_content
        ts = src['published'] / 1000
        day = datetime.datetime.fromtimestamp(timestamp=ts).weekday()
        rel_src['weekday'] = day

        snippets = []

        web_sim_terms = [] # type: List[str]
        query_snippets = ''

        rel_src['websim_found_name'] = 0

        for cls_n in cls_names:
            rel_src[cls_n + '_num_hl_terms'] = 0

        for x in (src['company_classifications'] + topic_clfs):
            cls_n = x['classifier_name']
            cls_conf = float(x['confidence'])
            rel_src[cls_n + '_clf_conf'] = 1 if cls_conf > 0 else 0
            rel_src[cls_n + '_num_hl_terms'] = len(
                x['hl_terms'] if x['hl_terms'] is not None else [])
            cls_snippets = None
            if x['snippets']:
                cls_snippets = eval(x['snippets'])

            if cls_n == 'websimilarity':
                if x['hl_terms']:
                    web_sim_terms = x['hl_terms']

            if cls_snippets and u'content.content_text' in cls_snippets:

                replace_em = lambda s: re.sub(r'<em>|</em>', '', s)
                for snip in cls_snippets[u'content.content_text']:
                    snippets.append(replace_em(snip))

                # For each snippet, calculate similarity to the title as proposed
                #  by Henning
                title = src['title']

                if cls_n == 'query':
                    query_snippets = ' '.join(
                        cls_snippets[u'content.content_text'])

        rel_src['num_snippets'] = len(snippets)

        count = 0
        for term in web_sim_terms:
            if term in query_snippets:
                count += 1

        rel_src['query_websim_intersects'] = count

        # Create features from word2vec
        # comp = '_'.join(re.split(split_pattern, replace_umlauts(
        #     comp_name).lower()))
        #
        # right_sims = []
        # if comp in right_w2v.vocab:
        #     cs = right_w2v.cosine(comp)
        #     right_sims = right_w2v.generate_response(*cs).tolist()
        #
        # wrong_sims = []
        # if comp in wrong_w2v.vocab:
        #     cs = wrong_w2v.cosine(comp)
        #     wrong_sims = wrong_w2v.generate_response(*cs).tolist()
        #
        # try:
        #     norm_factor_right = np.sum(np.array(right_sims)[:, 1])
        #     norm_factor_wrong = np.sum(np.array(wrong_sims)[:, 1])
        # except:
        #     norm_factor_right = 1
        #     norm_factor_wrong = 1
        #
        # split_content = set(re.split(split_pattern, content.lower()))
        # right_w2v_score = 0
        # wrong_w2v_score = 0
        #
        # # Count right weighted
        # for word, weight in right_sims:
        #     if word in split_content:
        #         right_w2v_score += weight
        #
        # # Count wrongs weighted
        # for word, weight in wrong_sims:
        #     if word in split_content:
        #         wrong_w2v_score += weight
        #
        # # Normalize counts
        # if norm_factor_right > 0:
        #     right_w2v_score /= norm_factor_right
        # if norm_factor_wrong > 0:
        #     wrong_w2v_score /= norm_factor_wrong
        #
        # rel_src['right_w2v_score'] = right_w2v_score
        # rel_src['wrong_w2v_score'] = wrong_w2v_score

        cls = 'evaluated'

        # Parse document to data matrix and target vector

        row = []
        for feat in features:
            row.append(rel_src[feat])
        data.append(row)
        target.append(rel_src[cls])
        corpus.append(rel_src['content'])

    # Transform to ndarray for encoding
    data = np.array(data)

    ensure_dir('./transformer/')

    # Scale first
    # data[:, 2:] = preprocessing.scale(data[:,2:].astype(float))

    # Encode nominal values to numeric values
    encode_companies(data, training)
    encode_domains(data, training)

    # Encode nominal values from above in a one-hot fashion
    data = one_hot_encode(data, training)
    data = csr_matrix(data)
    return data, target, corpus


def encode_companies(data, training):
    # type: (np.ndarray[np.ndarray[Union[int, float]]], bool) -> None
    """
    Encode companies in numeric values
    :param data: Input data
    :param training: If this is in the training stage or not
    :return:
    """
    companies = data[:, 0]
    if training:
        le_companies = LabelEncoder()
        le_companies.fit(np.append(companies, ['unknown']))
        pickle.dump(le_companies, open(
            './transformer/label_encoder_companies.p', 'wb'))
    else:
        le_companies = pickle.load(open(
            './transformer/label_encoder_companies.p', 'rb'))

    # If company is unseen set it to 'unknown'
    companies = [comp if comp in le_companies.classes_ else 'unknown'
                 for comp in companies]
    companies_encoded = le_companies.transform(companies)
    data[:, 0] = np.array(companies_encoded, dtype=int)


def encode_domains(data, training):
    # type: (np.ndarray[np.ndarray[Union[int, float]]], bool) -> None
    """
    Encode source domains in numeric values
    :param data: Input data
    :param training: If this is in the training stage or not
    :return:
    """
    source_domains = data[:, 1]
    if training:
        le_domains = LabelEncoder()
        le_domains.fit(np.append(source_domains, ['unknown']))
        pickle.dump(le_domains,
                    open('./transformer/label_encoder_domains.p', 'wb'))
    else:
        le_domains = pickle.load(open(
            './transformer/label_encoder_domains.p', 'rb'))

    # If domain is unseen set it to 'unknown'
    source_domains = [dm if dm in le_domains.classes_ else 'unknown'
                      for dm in source_domains]
    domains_encoded = le_domains.transform(source_domains)
    data[:, 1] = np.array(domains_encoded, dtype=int)


def one_hot_encode(data, training):
    # type: (np.ndarray[np.ndarray[Union[int, float]]], bool) -> np.ndarray[np.ndarray[Union[int, float]]]
    """
    Apply one hot encoding for companies and domains
    :param data: Input data
    :param training: If this is in the training stage or not
    :return:
    """
    if training:
        ohe = OneHotEncoder(categorical_features=[0, 1])

        # Simulate datapoint with unseen company and domain
        list0 = ['unknown', 'unknown'] # type: List[Union[str, int]]
        list1 = [0 for _ in range(0, data.shape[1] - 2)] # type: List[Union[str, int]]
        dummy_point = np.array([list0 + list1])
        encode_companies(dummy_point, training=False)
        encode_domains(dummy_point, training=False)
        data_with_dummy_point = np.append(data, dummy_point, axis=0)

        ohe.fit(data_with_dummy_point)
        pickle.dump(ohe, open('./transformer/one_hot_encoder.p', 'wb'))
    else:
        ohe = pickle.load(open('./transformer/one_hot_encoder.p', 'rb'))
    data = ohe.transform(data)
    return data


def get_data_with_content(path, s):
    # type: (str, int) -> Tuple[np.ndarray[np.ndarray[Any]],List[int], List[Any]]
    """
    Retruns the data from the given path
    :param path: data-path
    :param s: sampling percentage
    :return: X, y, corpus
    """

    # Check if data matrix has already been cached
    if os.path.isfile('./cached.p'):
        data = pickle.load(open('cached.p', 'rb'))
    else:
        data = parse_hits(docs=read(path), training=True, sampling=s)
        # data = filter(lambda x: x, data)
        pickle.dump(data, open('cached.p', 'wb'))
    return data


def delete_cache(type):
    # type: (str) -> None
    """
    Deletes the cache
    :param type: Can be 'data' or 'w2v'. If anything else, both (data and w2v
    cache will be deleted)
    :return:
    """
    data = False
    w2v = False

    if type == 'data':
        data = True
    elif type == 'w2v':
        w2v = True
    else:
        data = True
        w2v = True

    if data:
        remove_if_exists('cached.p')
    if w2v:
        remove_if_exists('right_content.txt')
        remove_if_exists('right_content_model.bin')
        remove_if_exists('right_content_phrases.txt')
        remove_if_exists('wrong_content.txt')
        remove_if_exists('wrong_content_model.bin')
        remove_if_exists('wrong_content_phrases.txt')


def remove_if_exists(file):
    # type: (str) -> None
    """
    Removes a file if it exists
    :param file:
    :return:
    """
    if os.path.isfile(file):
        os.remove(file)


def get_data(path, sampling=100):
    # type: (str, int) -> Tuple[np.ndarray[Union[float, int]], List[int]]
    """
    Gets the training data by path
    :param path: Path to data
    :param sampling: Sampling factor
    :return:
    """
    # build_word2vec_model(path, type='right', sampling=sampling)
    # build_word2vec_model(path, type='wrong', sampling=sampling)
    X, y, corpus = get_data_with_content(path, sampling)
    X = csr_matrix(X)
    return X, y


def get_data_from_elasticsearch(es, index):
    # type: (elasticsearch.Elasticsearch, str) -> Tuple[np.ndarray[np.ndarray[Any]],List[Any], List[Any]]
    """
    Gets the data from a given es-index
    :param es: es object
    :param index: es index
    :return: data
    """
    log.debug('Querying elasticsearch at {}'.format(index))
    response = es.search(index=index,
                         body={
                             "query": {
                                 "match_all": {}
                             },
                             "filter": {
                                 "bool": {
                                     "should": [
                                         {
                                             "term": {
                                                 "evaluated": "right"
                                             }
                                         },
                                         {
                                             "term": {
                                                 "evaluated": "wrong"
                                             }
                                         },
                                         {
                                             "term": {
                                                 "evaluated": "invalid"
                                             }
                                         }
                                     ]
                                 }
                             }
                         },
                         size=100000000 # Set this to 100.000.000
                         )

    return parse_hits(docs=response['hits']['hits'], training=True,
                      sampling=100)


def stem_corpus(corpus):
    # type: (Iterable[str]) -> Iterable[str]
    """
    Stems all words in a given corpus
    :param corpus: Corpus
    :return: Stemmed corpus
    """
    stemmer = snowballstemmer.stemmer('german')
    return map(lambda l: ' '.join(stemmer.stemWords(l.split())), corpus)


def ensure_dir(f):
    # type: (str) -> None
    """
    Ensures that a given directory exists
    :param f: directory path
    :return:
    """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
