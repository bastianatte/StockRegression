import pandas as pd
import numpy as np
import logging
import json
from sklearn.model_selection import TimeSeriesSplit
import sklearn


def load_config(path):
    """
    Load configuration file with all the needed parameters
    """
    with open(path, 'r') as conf_file:
        conf = json.load(conf_file)
    return conf

def get_logger(name):
    """
    Add a StreamHandler to a logger if still not added and
    return the logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = 1  # propagate to parent
        console = logging.StreamHandler()
        logger.addHandler(console)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
    return logger
utils_log = get_logger(__name__)
utils_log.setLevel(logging.INFO)


def load_single_csv(input_csv):
    """
    Creating df from single csv in a folder.
    """
    utils_log.info("in load single csv")
    df = pd.read_csv(input_csv)
    df_mod = df.loc[:, ('Date', 'Close', 'Open')]
    df_mod.sort_values('Date')
    df_mod.set_index('Date', inplace=True)        
    df_mod.loc[:,'Clop_norm'] = (df_mod.loc[:, 'Close'].subtract(df_mod.loc[:, 'Open'])).div(df_mod.loc[:, 'Open'])
    # utils_log.info("Dataset before applying lags: \n {}".format(df_mod.head()))
    # df_mod.loc[:, 'Clop_norm'] = df_mod.loc[:, 'Clop_norm'].shift(periods=1, fill_value=0)
    # utils_log.info("Dataset after applying lags: \n {}".format(df_mod.head()))
    days = 250
    train = df_mod.iloc[:-days].copy()
    test = df_mod.iloc[-days:].copy()
    X_train = train.loc[:, ('Close', 'Open')]
    utils_log.info("Before shift, Y_train: \n {}".format(train.loc[:, 'Clop_norm'].head()))
    y_train = train.loc[:, 'Clop_norm'].shift(periods=1, fill_value=0)
    # y_train.loc[:, 'Clop_norm'] = y_train.loc[:, 'Clop_norm'].shift(periods=1, fill_value=0)
    utils_log.info("After shift, Y_train: \n {}".format(y_train.head()))
    X_test = test.loc[:, ('Close', 'Open')]
    y_test = test.loc[:, 'Clop_norm']
    utils_log.info("shapes are: {}, {}, {}, {}".format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ))
    return df_mod, train, test, X_train, y_train, X_test, y_test


def calculate_sign(y_test, pred):
    good = 0 
    bad = 0
    for i, j in zip( range(len(y_test)), range(len(pred))):
        if np.sign(y_test[i]) == np.sign(pred[j]):
            good += 1
        else:
            bad += 1 
    print( "same sign:",(good/(good + bad))*100,"%, opposite sign: ", (bad/(good +bad))*100,"%")
    return (good/(good + bad))*100, (bad/(good + bad))*100