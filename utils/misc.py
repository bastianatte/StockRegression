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
    df_mod.loc[:,'actual_day_rt'] = (
        df_mod.loc[:, 'Close'].subtract(df_mod.loc[:, 'Open'])).div(df_mod.loc[:, 'Open']
        )
    utils_log.info("before nan: {}".format(df_mod.shape))
    df_mod.dropna(inplace=True)
    utils_log.info("after nan: {}".format(df_mod.shape))

    df_train = df_mod.loc['2005-01-01':'2015-01-01']
    df_test = df_mod.loc['2015-02-01':'2018-02-01']

    # train = df_mod.iloc[:-test_days].copy()
    # test = df_mod.iloc[-test_days:].copy()
    return df_mod, df_train, df_test

def split_df(df_full, train, test):
    """
    Splitting train and test dataframes in X and y.
    """
    # X_train = train.loc[:, ('Close', 'Open', 'Volume', 'High', 'Low')]
    # X_train = train.loc[:, ('Volume', 'High', 'Low')]
    X_train = train.loc[:, ('Close', 'Open')]
    y_train = train.loc[:, 'actual_day_rt'].shift(periods=-1, fill_value=0)
    # X_test = test.loc[:, ('Close', 'Open', 'Volume', 'High', 'Low')]
    # X_test = test.loc[:, ('Volume', 'High', 'Low')]
    X_test = test.loc[:, ('Close', 'Open')]
    y_test = test.loc[:, 'actual_day_rt'].shift(periods=-1, fill_value=0)

    df_check = test.copy()
    df_check['actual_next_day_rt'] = df_check.loc[:, 'actual_day_rt'].shift(periods=-1, fill_value=0)

    utils_log.info("shapes are: {}, {}, {}, {}".format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape
        ))
    return X_train, y_train, X_test, y_test, df_check

def building_df(y_test, next_day_return, str_stock, df_check):
    ndr = pd.Series(next_day_return, index=y_test.index, name='pred_next_day_rt')
    stocks_name = pd.Series(str_stock, index=y_test.index, name='ticker')
    result = pd.concat([y_test, ndr, stocks_name], axis=1, sort=False)
    result.reset_index(inplace=True)

    ## for testing purpose ###
    test = result.set_index(y_test.index)
    df_check[['pred_next_day_rt','ticker']]=result[['pred_next_day_rt', 'ticker']].to_numpy()
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

def define_periods(input_csv):
    utils_log.info("in define train test period")
    df = pd.read_csv(input_csv)
    df_mod = df.loc[:, ('Date')]
    df_mod_2 = df_mod.to_frame()
    utils_log.info("before nan: {}".format(df_mod.shape))
    df_mod.dropna(inplace=True)
    utils_log.info("after nan: {}".format(df_mod.shape))
    return df_mod_2

def hist_frequency_dates(df, df1, df2, out_path):
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    from datetime import datetime
    utils_log.info("in plot hist_frequency_dates")
    fig, ax = plt.subplots()
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    df["Date"] = df["Date"].astype("datetime64")
    df = pd.to_datetime(df["Date"])
    df1["Date"] = df1["Date"].astype("datetime64")
    df1 = pd.to_datetime(df1["Date"])
    df2["Date"] = df2["Date"].astype("datetime64")
    df2 = pd.to_datetime(df2["Date"])
    plt.hist(df, 1000, alpha=0.5, label='df full')
    plt.hist(df1, 1000, alpha=0.5, label='df train')
    plt.hist(df2, 1000, alpha=0.5, label='df test')
    ax.legend(loc='upper left', frameon=False)
    figname = os.path.join(out_path, "Date_periods" + ".png")
    plt.savefig(figname, dpi=200)
    plt.close()
    utils_log.info("plot hist_frequency_dates DONE!")

def create_folders(OUPUT_PATH):
    import os
    if not os.path.exists(OUPUT_PATH):
        os.makedirs(OUPUT_PATH)
    STOCKS_DIR = os.path.join(OUPUT_PATH, "Stocks")
    if not os.path.exists(STOCKS_DIR):
        os.makedirs(STOCKS_DIR)
    RANK_DIR = os.path.join(OUPUT_PATH, "Ranking")
    if not os.path.exists(RANK_DIR):
        os.makedirs(RANK_DIR)
    TABLE_DIR = os.path.join(RANK_DIR, "table")
    if not os.path.exists(TABLE_DIR):
        os.makedirs(TABLE_DIR)
    MISC_DIR = os.path.join(OUPUT_PATH, "misc")
    if not os.path.exists(MISC_DIR):
        os.makedirs(MISC_DIR)
    return STOCKS_DIR, RANK_DIR, TABLE_DIR, MISC_DIR

def create_each_stock_folder(arg_input, csv_file, STOCKS_DIR):
    import os
    path =str(arg_input)
    csv = os.path.join(path, csv_file)
    csv_folder_string = str(csv_file.strip(".csv"))
    CSV_DIR = os.path.join(STOCKS_DIR, csv_folder_string)
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
    return csv, CSV_DIR, csv_folder_string

def print_infos(cnfs):
    """
    Printing final general infos
    """
    if(cnfs["exe_lr"]):
        arith_mean(cnfs["lr_ss_list"], "lr same sign")
        arith_mean(cnfs["lr_os_list"], "lr opposite sign")
    if(cnfs["exe_rf"]):
        arith_mean(cnfs["rf_ss_list"], "rf same sign")
        arith_mean(cnfs["rf_os_list"], "rf opposite sign")
    if(len(cnfs['bad_df_list'])<=5):
        utils_log.info("less than 5 df with less than 1k entries")
    else:
        utils_log.info("{} dfs with less than 1K entries".format(len(cnfs['bad_df_list'])))
        for i, j in zip(range(len(cnfs['bad_df_list'])), range(len(cnfs['bad_df_shape_list']))):
            utils_log.info("BAD DFs: {}, with shape: {}".format(
                cnfs['bad_df_list'][i],
                cnfs['bad_df_shape_list'][j]))