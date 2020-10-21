import argparse
import fnmatch
import sys
import logging
import numpy
import os
from classes.models import models
from classes.plotter import plotter
from utils.plots import (
    linear_regression_plot, linear_regression_scatter, plot_sign
    )
from utils.misc import (
    get_logger, load_config, load_single_csv, calculate_sign
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
logger.info("Reading settings")
OUPUT_PATH = args.output

i = 0
for csv_file in os.listdir(args.input):
    if fnmatch.fnmatch(csv_file, '*.csv'):
    	i += 1
    	if i == 20:
    		break
    	else:
    		logger.info("##################")
    		path =str(args.input)
	    	logger.info("{}! Reading {} file ".format(i, csv_file))
	    	logger.info("Joining path({}) and csv({})".format(path, csv_file))
	    	csv = os.path.join(path, csv_file)
	    	logger.info("csv: {}".format(csv))
	    	csv_folder_string = str(csv_file.strip(".csv"))
	    	CSV_DIR = os.path.join(OUPUT_PATH, csv_folder_string)
	    	if not os.path.exists(CSV_DIR):
	    		os.makedirs(CSV_DIR)
	    	logger.info("csv dir: {}".format(CSV_DIR))

	    	# loading single csv ###
	    	df, train, test, X_train, y_train, X_test, y_test = load_single_csv(csv)

	    	## plot ##
	    	globals()[csv_folder_string] = plotter(df, train, test, CSV_DIR)
	    	globals()[csv_folder_string].plot_train_test()
	    	globals()[csv_folder_string].plot_close_open()

	    	# loading models ###
	    	globals()[csv_folder_string] = models(train, test,
	    		X_train, y_train,
	    		X_test, y_test,
	    		CSV_DIR)

	    	# ### linear regression model ###
	    	lm, next_day_return = globals()[csv_folder_string].linear_regression()
	    	print(next_day_return[:3], y_test[:3])
	    	linear_regression_plot(y_test, next_day_return, CSV_DIR)
	    	# plot_sign(y_test, train_pred, CSV_DIR)


