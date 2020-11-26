from utils.misc import get_logger
import logging

rank_log = get_logger(__name__)
rank_log.setLevel(logging.INFO)

def exe_rank(series, path, plot_out_path):
	import pandas as pd
	from datetime import datetime
	import os

	print(series.shape, series.columns)
	short_filename = os.path.join(path, r'short.csv')
	short_part_next_df = pd.DataFrame()
	long_part_next_df = pd.DataFrame()
	long_short_nocost_accu_profit = 0
	long_short_profit_nocost = {}
	long_short_accuprofit_nocost = {}
	long_daily_profit_dict=[]
	short_daily_profit_dict=[]
	cumul_profit = []
	k = 10
	SYMBOL = 'Title'
	long_share = 0.5
	short_share = 0.5

	### series snippet ###
	series = series.sort_values(['Date', 'pred_next_day_rt'], ascending=[True, False])
	date_list = list(series.Date.unique())
	date_list.sort()
	i = 0
	print(len(date_list))
	for date in date_list[:-1]:
		if i == 0:
			date_string = str(date)
			print("#####@@@@@@@@@@@@@@@######", date_string)
		i += 1
		data_date = series[series.Date == date]
		data_date = data_date.drop_duplicates(subset=['ticker'])
		long_part = data_date.iloc[:k]
		short_part = data_date.iloc[-k:]

		long_df = long_part
		short_df = short_part
		print(long_df)
		long_daily_profit = (long_df['actual_day_rt']).mean()
		long_daily_profit_dict.append(long_daily_profit)
		short_daily_profit = ((-1)*short_df['actual_day_rt']).mean()
		short_daily_profit_dict.append(short_daily_profit)
		rank_log.info("\n ####### \n LONG PART: \n {}".format(long_df))
		rank_log.info("\n SHORT PART: \n {} \n #######".format(short_df))
		daily_profit = long_share*long_daily_profit + short_share*short_daily_profit
		long_short_nocost_accu_profit += daily_profit
		long_short_profit_nocost.update({date:daily_profit})
		long_short_accuprofit_nocost.update({date:long_short_nocost_accu_profit})
		date_obj = datetime.strptime(date, '%Y-%M-%d')
		cumul_profit.append({'Date': date_obj, 'cum_profit': long_short_nocost_accu_profit})
	plot_long_short_daily_profit(long_daily_profit_dict, short_daily_profit_dict,
		long_short_profit_nocost,
		plot_out_path)
	plot_daily_profit(long_short_profit_nocost, plot_out_path)
	hist_daily_profit(long_short_profit_nocost, plot_out_path)
	plot_long_short_nocost_accu_profit(long_short_accuprofit_nocost, date_string, plot_out_path)
	plot_long_short_nocost_accu_profit_sns(long_short_accuprofit_nocost, date_string, plot_out_path)
	return long_short_profit_nocost, long_short_accuprofit_nocost, long_part_next_df, short_part_next_df


def creating_ranking_csv(long_short_profit_nocost, long_part_next_df, short_part_next_df, path):
	import pandas as pd
	import os

	### csv filenames ###
	file_name_rntc = os.path.join(path, r'returns.csv')
	file_name_rntcs = os.path.join(path, r'returns_series.csv')
	long_filename = os.path.join(path, r'long.csv')
	short_filename = os.path.join(path, r'short.csv')
	### return no transaction cost
	returns_no_transaction_cost = pd.DataFrame(columns=['Returns'], 
		data=long_short_profit_nocost.values(), 
		index=long_short_profit_nocost.keys()
		)
	print(returns_no_transaction_cost)
	returns_no_transaction_cost.index.name = 'Date'
	returns_no_transaction_cost = returns_no_transaction_cost[returns_no_transaction_cost!=0]
	returns_no_transaction_cost = returns_no_transaction_cost.dropna()
	# print(returns_no_transaction_cost)
	# print("retun index name: ", returns_no_transaction_cost.index.name)

	### return no transaction cost series
	s = pd.Series(long_short_profit_nocost, name='Returns')
	s.index.name = 'Date'
	s.reset_index()
	s.columns = ['Date', 'Returns']
	s = s[s!=0]

	### store csv files for further analysis ###
	returns_no_transaction_cost.to_csv(file_name_rntc)
	s.to_csv(file_name_rntcs)
	long_part_next_df.to_csv(long_filename)
	short_part_next_df.to_csv(short_filename)
	return s

def plot_cumulative(lspncs, out_path):
	import numpy as np
	import matplotlib.pyplot as plt
	import os
	cumulative = np.cumsum(lspncs)
	plt.plot(cumulative, c='blue')
	plt.plot(len(cumulative)-cumulative, c='green')
	figname = os.path.join(out_path, "cumulative"+".png")
	plt.savefig(figname, dpi=200)
	plt.close()

def plot_daily_profit(long_short_profit_nocost, out_path):
	import matplotlib.pyplot as plt
	import os
	plt.plot(long_short_profit_nocost.values(), ':', c='blue')
	plt.title("daily profit")
	figname = os.path.join(out_path, "plot_daily_profit"+".png")
	plt.savefig(figname, dpi=200)
	plt.close()

def hist_daily_profit(long_short_profit_nocost, out_path):
	import matplotlib.pyplot as plt
	import os
	plt.hist(long_short_profit_nocost.values(), 100, alpha=0.5, label='daily profit')
	plt.title("daily profit")
	figname = os.path.join(out_path, "hist_daily_profit"+".png")
	plt.savefig(figname, dpi=200)
	plt.close()


def plot_long_short_nocost_accu_profit(long_short_accuprofit_nocost, date_string, out_path):
	import matplotlib.pyplot as plt
	import os
	plt.plot(long_short_accuprofit_nocost.values(), ':', c='blue', label=date_string)
	plt.title("Cumulative")
	plt.legend(loc="upper right")
	figname = os.path.join(out_path, "cumulative_" + ".png")
	plt.savefig(figname, dpi=200)
	plt.close()

def plot_long_short_nocost_accu_profit_sns(long_short_accuprofit_nocost, date_string, out_path):
	import seaborn as sns
	import os
	# print(long_short_accuprofit_nocost.keys())
	plot = sns.lineplot(data=long_short_accuprofit_nocost.values(), markers=True,linewidth=2, markersize=10)
	figname = os.path.join(out_path, "sns_cumulative_" + ".png")
	fig = plot.get_figure()
	fig.savefig(figname)

def plot_long_short_daily_profit(long_daily_profit, short_daily_profit, long_short_profit_nocost, out_path):
	import matplotlib.pyplot as plt
	import os
	plt.hist(long_daily_profit, 100, alpha=0.5, label='long daily profit')
	plt.hist(short_daily_profit, 100, alpha=0.5, label='short daily profit')
	plt.hist(long_short_profit_nocost.values(), 100, alpha=0.5, label='daily profit')
	plt.title("long short daily profit")
	plt.legend(loc="upper left")
	figname = os.path.join(out_path, "full_daily_profit"+".png")
	plt.savefig(figname, dpi=200)
	plt.close()	