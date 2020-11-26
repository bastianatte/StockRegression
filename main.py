import argparse
import fnmatch
import sys
import logging
import numpy
import pandas as pd
import os
import json
from classes.models import models
import pandas as pd
from classes.plotter import plotter
from utils.plots import (
    linear_regression_plot, linear_regression_scatter, plot_sign
    )
from utils.misc import (
    get_logger, load_config, load_single_csv, create_folders, calculate_sign, arith_mean,
    split_df, building_df, define_periods, hist_frequency_dates, create_each_stock_folder,
    print_infos
	)
from utils.ranking import (
	exe_rank, creating_ranking_csv, plot_cumulative, plot_daily_profit,
	plot_long_short_nocost_accu_profit
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
### declare logger ###
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

### config file ####
logger.info("Reading config file")
args = parser.parse_args()
with open(args.conf) as json_data_file:
    CONF = json.load(json_data_file)
STOCKS_DIR, RANK_DIR, TABLE_DIR, MISC_DIR = create_folders(args.output)


### loop over stocks dfs ###
i = 0
for csv_file in os.listdir(args.input):
    if fnmatch.fnmatch(csv_file, '*.csv'):
    	if i == 5:
    		break
    	else:
    		i += 1
	    	csv, CSV_DIR, csv_folder_string = create_each_stock_folder(str(args.input), csv_file, STOCKS_DIR)
	    	# loading single csv ###
	    	df, df_train, df_test = load_single_csv(csv, 250)
	    	if len(df.index) < 2000:
	    		CONF['main']['bad_df_list'].append(csv_file)
	    		CONF['main']['bad_df_shape_list'].append(df.shape)	 
	    		continue
	    	df_full = define_periods(csv)
	    	CONF["main"]["df_full_list"].append(df_full)
	    	CONF["main"]["df_train_list"].append(df_train)
	    	CONF["main"]["df_test_list"].append(df_test)
	    	X_train, y_train, X_test, y_test, df_check = split_df(df, df_train, df_test)

	    	# plot ##
	    	if (CONF["main"]["enable_plots_per_stocks"]):
		    	ppplltt = plotter(df, df_train, df_test, CSV_DIR)
		    	ppplltt.plot_train_test()
		    	ppplltt.plot_close_open()
		    	ppplltt.plot_actual_day_rt()

# 	    	### loading models ###
	    	mod = models(X_train, y_train, X_test, y_test)
	    	### random forest regressor model ###
	    	if(CONF["main"]["exe_rf"]):
	    		rfm, next_day_return_rf = mod.random_forest()
		    	rf_ss, rf_os = calculate_sign(y_test, next_day_return_rf)
		    	CONF["main"]["rf_ss_list"].append(rf_ss)
		    	CONF["main"]["rf_os_list"].append(rf_os)
	    	### linear regression model ###
	    	if(CONF["main"]["exe_lr"]):
	    		lm, next_day_return_lr = mod.linear_regression()
		    	linear_regression_plot(y_test, next_day_return_lr, CSV_DIR)
		    	plot_sign(y_test, next_day_return_lr, CSV_DIR)
		    	lr_ss, lr_os = calculate_sign(y_test, next_day_return_lr)
		    	CONF["main"]["lr_ss_list"].append(lr_ss)
		    	CONF["main"]["lr_os_list"].append(lr_os)


	    	### append next day return to y_test ###
	    	# single_df = building_df(y_test, next_day_return_lr, csv_folder_string, df_check)
	    	single_df = building_df(y_test, next_day_return_rf, csv_folder_string, df_check)
	    	CONF["main"]["series_list"].append(single_df)







df_full_dates = pd.concat(CONF["main"]["df_full_list"])
df_train_dates = pd.concat(CONF["main"]["df_train_list"])
df_test_dates = pd.concat(CONF["main"]["df_test_list"])
hist_frequency_dates(df_full_dates, df_train_dates, df_test_dates, MISC_DIR)

full_df = pd.concat(CONF["main"]["series_list"])
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
print("##########@@@@@@@@@@@@@@@@@~~~~~~~", full_df)

###### RANKING ######
long_short_profit_nocost, long_short_nocost_accu_profit, long_part_next_df, short_part_next_df = exe_rank(full_df, TABLE_DIR, RANK_DIR)
lspncs = creating_ranking_csv(long_short_profit_nocost, long_part_next_df, short_part_next_df, TABLE_DIR)

print_infos(CONF["main"])
