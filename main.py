import argparse
import fnmatch
import sys
import logging
import numpy
import os
from classes.models import models
import pandas as pd
from classes.plotter import plotter
from utils.plots import (
    linear_regression_plot, linear_regression_scatter, plot_sign
    )
from utils.misc import (
    get_logger, load_config, load_single_csv, calculate_sign, arith_mean, split_df,
    building_df
)
from utils.ranking import long_short_rank


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
lr_ss_list = []
lr_os_list = []
rf_ss_list = []
rf_os_list = []
series_list = []
bad_df_list = []
bad_df_shape_list = []
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
	    	logger.info("csv input dir: {}".format(csv))
	    	csv_folder_string = str(csv_file.strip(".csv"))
	    	CSV_DIR = os.path.join(OUPUT_PATH, csv_folder_string)
	    	if not os.path.exists(CSV_DIR):
	    		os.makedirs(CSV_DIR)
	    	logger.info("csv output dir: {}".format(CSV_DIR))

	    	# loading single csv ###
	    	df, train, test = load_single_csv(csv, 250)
	    	if len(df.index) < 1000:
	    		bad_df_list.append(csv_file)
	    		bad_df_shape_list.append(df.shape)
	    		continue
	    	X_train, y_train, X_test, y_test = split_df(train, test)

	    	## plot ##
	    	globals()[csv_folder_string] = plotter(df, train, test, CSV_DIR)
	    	# globals()[csv_folder_string].plot_train_test()
	    	# globals()[csv_folder_string].plot_close_open()

	    	# loading models ###
	    	globals()[csv_folder_string] = models(train, test,
	    		X_train, y_train,
	    		X_test, y_test,
	    		CSV_DIR)

	    	### linear regression model ###
	    	lm, next_day_return_lr = globals()[csv_folder_string].linear_regression()
	    	# linear_regression_plot(y_test, next_day_return, CSV_DIR)
	    	# plot_sign(y_test, train_pred, CSV_DIR)
	    	lr_ss, lr_os = calculate_sign(y_test, next_day_return_lr)
	    	lr_ss_list.append(lr_ss)
	    	lr_os_list.append(lr_os)

	    	### random forest classifier ###
	    	rfm, next_day_return_rf = globals()[csv_folder_string].random_forest()
	    	rf_ss, rf_os = calculate_sign(y_test, next_day_return_rf)
	    	rf_ss_list.append(rf_ss)
	    	rf_os_list.append(rf_os)

	    	### append next day return to y_test ###
	    	# single_df = building_df(y_test, next_day_return_lr, csv_folder_string)
	    	single_df = building_df(y_test, next_day_return_rf, csv_folder_string)	    	
	    	series_list.append(single_df)
full_df = pd.concat(series_list)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(full_df)

#########################################
arith_mean(lr_ss_list, "lr same sign")
arith_mean(lr_os_list, "lr opposite sign")
arith_mean(rf_ss_list, "rf same sign")
arith_mean(rf_os_list, "rf opposite sign")
for i, j in zip(range(len(bad_df_list)), range(len(bad_df_shape_list))):
	logger.info("BAD DFs: {}, {}".format(bad_df_list[i], bad_df_shape_list[j]))
############################################

###### RANKING ######
long_short_rank(full_df)



