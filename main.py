import argparse
import sys
import logging
import numpy
import os
from classes.models import models
from classes.plotter import plotter
from utils.plots import check_stationarity
from utils.misc import (
    get_logger, load_config, load_data, load_single_csv
)


parser = argparse.ArgumentParser(description=("Load flaf to run StockRegression"))
parser.add_argument('-c', '--conf', type=str, metavar='', required=True,
    help='Specify the path of the configuration file'
    )
parser.add_argument('-i', '--input', type=str, metavar='', required=True,
    help='Specify the path of the root file'
)
parser.add_argument('-o', '--output', type=str, metavar='', required=True,
    help='Specify the ouput path'
)
args = parser.parse_args()
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
try:
    CONF = load_config(args.conf)
    MAIN = CONF.get("main")
except Exception as e:
    logger.error("{}. could not get conf settings".format(e.args[0]))
    sys.exit(1)
logger.info("Reading settings")
OUPUT_PATH = args.output
PLOT_DIR = os.path.join(OUPUT_PATH, "Plots")
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

######### reading dataset #########
# feat, df_full = load_data(args.input)
df, df_train, df_test= load_single_csv(args.input)

######### plot dataframe ##########
plotter = plotter(df, df_train, df_test, PLOT_DIR)
plotter.plot_clop()
plotter.plot_clop_norm()
plotter.plot_close_open()
plotter.plot_train_test()
plotter.check_stationarity()
######## models ##############
mod = models(df_train, df_test, PLOT_DIR)
# mod.arima_test()
mod.linear_regression()
