def long_short_rank(series):
	import pandas as pd
	series = series.sort_values(['Date', 'next_day_rt'], ascending=[True, False])
	date_list = list(series.Date.unique())
	date_list.sort()
	k = 10
	SYMBOL = 'Title'
	long_share = 0.5
	short_share = 0.5

	### to datetime for checking reasons ##
	series.Date = pd.to_datetime(series.Date)

	print("printing series Date: \n ", series.Date)
	match_daystamp = "2018-12-18"
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
		long_short_nocost_accu_profit = 0
		long_short_profit_nocost = {}
		long_short_accuprofit_nocost = {}
		long_short_nocost_accu_profit += daily_profit
		long_short_profit_nocost.update({date:daily_profit})
		long_short_accuprofit_nocost.update({date:long_short_nocost_accu_profit})
		# print("ls p nc: ", long_short_profit_nocost, "---, ls ap nc: ", long_short_accuprofit_nocost)

		date_index = date_list.index(date)
		long_part_next = series[series.Date == date_list[date_index+1]].iloc[:k]
		short_part_next = series[series.Date == date_list[date_index+1]].iloc[-k:]
		print("##############")
		print("LONG PART: \n",long_part_next)
		print("SHORT PART: \n", short_part_next)
		print("##############")


		returns_no_transaction_cost = pd.DataFrame(columns=['Returns'], 
			data=long_short_profit_nocost.values(), 
			index=long_short_profit_nocost.keys())