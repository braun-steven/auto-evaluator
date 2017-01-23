"""
Config options:

    host: <es_host>                 See below:
    port: <es_port>                 Connect elasticsearch on <es_host>:<es_port>
    rest_port: <application_port>   Listen on <application_port>.
    logfile: <filename>             Write log to <filename>.
    loglevel: <level>               Log log-level <level> (default: INFO).
    debug: <debug>                  Run application in debug mode (True/False).
    crawl_index: <index_name>       index containing crawled websites
    crawl_type: <doc_type>          crawl - doc_type
    min_score: <min_score>          Min es-similarity score
    top_n: <n>                      Top n docs to retrieve
    like_fields: <fields>           like_fields

Note: the configuration format is json.
"""
import os
import sys
import json
import logging

default_config = dict(
    train_index="proposal-training",
    n_jobs=4,
    model_path="model.p",
    threshold=0.25,
    host="localhost",
    logfile="./eval.log",
    loglevel="DEBUG",
    port=9999,
    rest_port=5099,
    tune_params=0,
    retrain=0,
    streamhandler_config_path="./streamhandler.config.json"
)


def read_config(fname=None):
    """Read the config from file (if filename given)
    and merge with default values.
    """
    if fname is None or not os.path.exists(fname):
        print("Config file {} not found! Using defaults.".format(fname))
        return default_config

    with open(fname) as f:
        config = json.load(f)

    # Merge default_config and config:
    return {key: config.get(key) or default_config.get(key) for key in
                     set(default_config) | set(config)}

def write_config(fname, config):
    """Write config file to fname.
    """
    with open(fname, 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)
    print(__doc__)


def configure_logging(config):
    """Configures the logging module.
    """
    logfile = config.get('logfile')
    loglevel = config.get('loglevel') or 'INFO'
    if logfile is not None:
        logging.basicConfig(filename=logfile, level=loglevel,
                            format='%(asctime)s: %(name)s:%(levelname)s:: %(message)s')
    else:
        print "Setting up stdout logger..."
        logging.basicConfig(stream=sys.stdout, level=loglevel,
                            format='%(name)s:%(levelname)s:: %(message)s')
    logging.getLogger('elasticsearch').setLevel('WARNING')
    logging.getLogger('urllib3').setLevel('WARNING')
