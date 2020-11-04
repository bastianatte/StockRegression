from utils.misc import get_logger
import logging

rank_log = get_logger(__name__)
rank_log.setLevel(logging.INFO)

def long_short_rank(series, path):
	import pandas as pd
	import os
	short_filename = os.path.join(path, r'short.csv')
	short_part_next_df = pd.DataFrame()
	long_part_next_df = pd.DataFrame()
	long_short_nocost_accu_profit = 0
	long_short_profit_nocost = {}
	long_short_accuprofit_nocost = {}
	k = 10
	SYMBOL = 'Title'
	long_share = 0.5
	short_share = 0.5

	### series snippet ###
	series = series.sort_values(['Date', 'next_day_rt'], ascending=[True, False])
	date_list = list(series.Date.unique())
	date_list.sort()
	for date in date_list[:-1]:
		data_date = series[series.Date == date]
		data_date = data_date.drop_duplicates(subset=['stoke_name'])
		long_part = data_date.iloc[:k]
		short_part = data_date.iloc[-k:]
		long_df = long_part
		short_df = short_part
		long_daily_profit = (long_df['Clop_norm']).mean()
		short_daily_profit = ((-1)*short_df['Clop_norm']).mean()
		daily_profit = long_share*long_daily_profit + short_share*short_daily_profit
		long_short_nocost_accu_profit += daily_profit
		long_short_profit_nocost.update({date:daily_profit})
		long_short_accuprofit_nocost.update({date:long_short_nocost_accu_profit})

		date_index = date_list.index(date)
		long_part_next = series[series.Date == date_list[date_index+1]].iloc[:k]
		short_part_next = series[series.Date == date_list[date_index+1]].iloc[-k:]
		rank_log.info("\n ####### \n LONG PART: \n {}".format(long_part_next))
		rank_log.info("SHORT PART: \n {} \n #######".format(short_part_next))
		long_part_next_df = long_part_next_df.append(long_part_next)
		short_part_next_df = short_part_next_df.append(short_part_next)

	return long_short_profit_nocost, long_part_next_df, short_part_next_df



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
	returns_no_transaction_cost.index.name = 'Date'
	print(returns_no_transaction_cost)
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
