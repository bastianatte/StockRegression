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


def load_single_csv(input_csv, test_days):
    """
    Creating df from single csv in a folder.
    """
    utils_log.info("in load single csv")
    df = pd.read_csv(input_csv)
    df_mod = df.loc[:, ('Date', 'Close', 'Open', 'Volume','High', 'Low')]
    df_mod.sort_values('Date')
    df_mod.set_index('Date', inplace=True)
    df_mod.loc[:,'Clop_norm'] = (
        df_mod.loc[:, 'Close'].subtract(df_mod.loc[:, 'Open'])).div(df_mod.loc[:, 'Open']
        )
    utils_log.info("before nan: {}".format(df_mod.shape))
    df_mod.dropna(inplace=True)
    utils_log.info("after nan: {}".format(df_mod.shape))
    train = df_mod.iloc[:-test_days].copy()
    test = df_mod.iloc[-test_days:].copy()
    return df_mod, train, test

def split_df(train, test):
    """
    Splitting train and test dataframes in X and y.
    """
    # X_train = train.loc[:, ('Close', 'Open', 'Volume', 'High', 'Low')]
    X_train = train.loc[:, ('Close', 'Open')]
    y_train = train.loc[:, 'Clop_norm'].shift(periods=1, fill_value=0)
    # X_test = test.loc[:, ('Close', 'Open', 'Volume', 'High', 'Low')]
    X_test = test.loc[:, ('Close', 'Open')]
    y_test = test.loc[:, 'Clop_norm']
    utils_log.info("shapes are: {}, {}, {}, {}".format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ))
    return X_train, y_train, X_test, y_test

def building_df(y_test, next_day_return, str_stoke):
    ndr = pd.Series(next_day_return, index=y_test.index, name='next_day_rt')
    # print(ndr)
    stokes_name = pd.Series(str_stoke, index=y_test.index, name='stoke_name')
    result = pd.concat([y_test, ndr, stokes_name], axis=1, sort=False)
    result.reset_index(inplace=True)
    utils_log.info("df columns are: {}".format(result.columns))
    # print(result)
    return result

def calculate_sign(y_test, pred):
    good = 0 
    bad = 0
    for i, j in zip( range(len(y_test)), range(len(pred))):
        if np.sign(y_test[i]) == np.sign(pred[j]):
            good += 1
        else:
            bad += 1 
    good_per = (good/(good + bad))*100
    bad_per = (bad/(good +bad))*100
    utils_log.info("same sign: {}%, opposite sign: {}%".format(good_per, bad_per))
    return good_per, bad_per

def arith_mean(list, string):
    val_sum=0
    idx = 0 
    val = 0
    for idx, val in enumerate(list):
        val_sum += val
        # utils_log.info("index is: {}, and {} values are: {}".format(idx, string, val))
    am = val_sum/(idx+1)
    utils_log.info("Arith mean {} list is: {}%".format(string, am))

