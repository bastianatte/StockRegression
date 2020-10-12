import pandas as pd
import logging
import json


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

def load_data(input_csv):
    """
    Creating dataset from multiple csv file in a folder
    """
    import numpy
    # dataset = pd.read_csv(input_csv,  parse_dates=['Date'], index_col='Date')
    dataset = pd.read_csv(input_csv)
    header = dataset.columns
    features = dataset.loc[:, dataset.columns].values
    print("feat type: ", type(features))
    utils_log.info("dataset headers: \n {}".format( header ))
    utils_log.info("Dataset head: \n {}".format(dataset.head))
    utils_log.info("features shape : {}".format(features.shape))
    return features, dataset

def load_single_csv(input_csv):
    """
    Creating df from single csv in a folder.
    """
    df = pd.read_csv(input_csv)
    df_mod = df[['Date','Open',"Close"]]
    df_mod.sort_values('Date', inplace=True)
    df_mod.set_index('Date', inplace=True)        
    utils_log.info("df before: \n {}".format(df_mod))
    df_mod['Clop'] = df_mod['Close'] - df_mod['Open']
    df_mod['Clop_norm'] = df_mod['Clop']/df_mod['Open']
    print("#####################")
    print(df_mod)
    print("#####################")
    #####################
    days = 250
    train = df_mod.iloc[:-days,3:4].copy()
    test = df_mod.iloc[-days:,3:4].copy()
    utils_log.info("train shape is {} and test shape is {}".format(
        train.shape, test.shape
        ))
    utils_log.info("Train df: {} \n Test df: {}".format(train.head, test.head))
    return df_mod, train, test


