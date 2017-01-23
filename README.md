# Automatic article evaluation

This module contains the evaluation classifier. It implements a machine learning based approach to learn pattern and recognize whether an article should be evaluated as 'wrong' or  not. 

## How to use the classifier
```
$ python2 classifier_eval.py -h
Usage:
    classifier_eval -h
    classifier_eval -c <config>

Options:
    -h, --help                                  Show this screen.
    -c <config>, --config <config>              Use configfile <config>
```

The configuration is as follows:

```
{
  "train_index": "proposal-training",   # es-index of the trainin data
  "n_jobs": 4,                          # number of parallel jobs 
  "model_path": "model.p",              # path at which the model shall be stored/loaded
  "threshold": 0.25,                    # threshold as described below
  "host": "elasticsearch",              # es-host
  "logfile": "/log/eval.log",           # logfile
  "loglevel": "DEBUG",                  # loglevel
  "port": 9200,                         # es-port
  "rest_port": 5099,                    # REST-port for this classifier,
  "tune_params":0,                      # Starts a gridsearch on training over an interval of different parameters for each classifier
  "streamhandler_config_path": 
         "./streamhandler.config.json"  # Path to the configuration-file of the streamhandler
}
```


## Theoretical background

The actual classification task is done by several internal algorithms which are currently straight forward combined to an ensemble of classifiers. The ensemble can be set up to do a weighted vote (the average over all confidences) or a max vote (the prediction of the classifier with the highest confidence) over the single predictions. The following algorithms are used in the ensemble:

* [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier): Naive approach which assumes feature independence
* [Random Forest](https://en.wikipedia.org/wiki/Random_forest): An ensemble of decision trees using bagging and bootstrapping
* ~~[Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent): Stochastic approximation for the gradient descent optimization problem~~ 
(removed)
* [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression): Builds a linear combination of the features (-variables) to predict the label
* [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine): Solves a hyperplane margin optimization problem

## Technical description

### Modules

* `classifier_eval` contains the actual classifier as microservice
* `my_ensembles` define the underlying classifier in the [standard sklearn fashion](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator)
* `data_helper` provides parsing methods which are used in the classifier
* `evaluator`, `main` and `testplot` are modules for the testing environment

### How to setup the classifier

The classifier can be instantiated as follows:
```python
config = {
  ...
  "threshold": 0.25,
  ...
}

clf = EvalClassifier(config)
```
The threshold is a limit at which the classifier decides wether the predicted should be a 'wrong' or not. The actual output for an article looks like `(0.85, 0.15)`, which means it has a confidence of `0.85` that the output is labeled as 'wrong'. The threshold refers to the second value, this means if the second value (confidence for the article not being a 'wrong') is below the threshold, the classifier predicts 'wrong'. If the second value is above the threshold, the classifier predicts 'not wrong' (namely 'right/invalid').

Since this classifier is based on an machine learning approach it needs to build up a so called model, which contains the generalized pattern found in the training data. 

The training data currently needs to be in the format of an ElasticSearch query result. This means that all training data can be put into a single index and the result of a query on the whole index can be used to train the classifier.
```python
clf.build()
```

After the training has finished the classifier can be serialized with:

```python
clf.save_classifier(save_path)
```

If the classifier should not be retrained on the data and a model has already been stored, it can be loaded with:

```python
clf.load_classifier(load_path)
```

### How to predict the evaluation of articles

The input format for the classification task is the same as for the training step. The results of an ES-Query can be used and will be transformed to fit for the classifier. Simply call

```python
clf.classify(docs)
```

## Enhancing the classifier

The power of this classifier depends heavily on the features which are built from the input data. To improve the classifier one can simply [create new features](http://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/). Lets assume we want to add a new (dumb) feature which takes into account the length of the articles content namely `content_length`:

In the method `parse_hits(...)` the list of features needs to be extended:
```python
features = [
        'company_uid',
        'publisher',
        ...
        'content_length' # <-- Adds new feature name
    ]
```
Attention: Do not touch the first two elements of the features list, since they will be encoded later on. 

The variable `src` contains all fields of the current document in the for loop. We now need to get the content and add its length to the relevant source dictionary `rel_src`:

```python
content = src['content']
cl = len(content)
rel_src['content_length'] = cl
```

Thats it! You should now evaluate the classifier with the new set of features


## Evaluating the classifier

The module `main` can be executed with the following options:

```
$ python2 main.py -h
usage: main.py [-h] [-d D] [-clf CLF] [-t T] [-clean CLEAN] [-s S]
               [-n_jobs N_JOBS]

Apply machine learning on article evaluation

optional arguments:
  -h, --help      show this help message and exit
  -d D            Path to the data
  -clf CLF        Classifiertype (May be "gb", "svm", "nb" or "rf")
  -t T            Specifies a tag for the output
  -clean CLEAN    Deletes the cached dataset
  -s S            Sampling rate in percent
  -n_jobs N_JOBS  Number of parallel jobs if available

```

An example could be:

```
$ python2 main.py -d data-proposal.json -clf eoc -s 20 -n_jobs 4 -clean true
```

This loads all data from `data-proposal.json`, uses the standard Ensemble of Classifiers as estimator, samples from the data with 20%, runs the training process in parallel on 4 cores if possible and removes old cache files. 


### Evaluation output
The results of the evaluation can be found in the folder `vis` and the according subfolder for the classifier which has been built. The file `out.log` shows some information about the evaluation process like:

```
Arguments: Namespace(clean='true', clf='eoc', d='data-proposal.json', n_jobs='4', s='100', t=None)
Chosen classifier: 
EOCClassifier(aggregate='weighted', n_jobs=1, tune_parameters=None)
Classification report:              precision    recall  f1-score   support

      wrong       0.86      0.86      0.86     17289
      right       0.91      0.91      0.91     26828

avg / total       0.89      0.89      0.89     44117

Accuracy score: 0.891334406238
Log loss: 0.284254393254
Confusion matrix, without normalization
[[14843  2446]
 [ 2348 24480]]
Normalized confusion matrix
[[ 0.86  0.14]
 [ 0.09  0.91]]
```

The actual goal of the classifier is to minimize the false negatives (true label: positive, predicted label: negative, currently 9%) and maximize the true negatives (true label: negative, predicted label: negative, currently 86%). This task is equivalent to maximizing the AUC for the ROC-Curve. The ROC-Curve, aswell as the Precision-Recall-Curve and the confusion matrix (normalized and absolut) can be found in `eval.png`. 

### How to choose a good threshold?

The plot in `cm_ths.png` shows specificity, sensitivity, miss-rates and fall-out for all prediction probability thresholds between 0 and 1. The red line shows how the false negative rate and the blue line how the true negative rate increase with higher thresholds. Red means 'risk' for loosing a good article and blue means 'efficiency' in the filtering process of unnecessary (wrong) articles. 

![Metrics ](https://github.com/slang03/auto-evaluator/blob/master/examples/cm_ths.png)

## Copyright
The files are intellectual property of [PRIME Research](http://prime-research.com/en/)
