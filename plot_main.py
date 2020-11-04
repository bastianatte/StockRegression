import pandas as pd
import matplotlib.pyplot as plt
import argparse
import csv
import os
import pandas_datareader as web
from datetime import datetime
import seaborn as sns
import matplotlib.dates as mdates
import pytz
import pyfolio as pf

parser = argparse.ArgumentParser(description=("Load csv to plot StockRegression control plots"))
parser.add_argument('-i', '--input', type=str, metavar='', required=True,
    help='Specify the path of csv file'
)
parser.add_argument('-o', '--output', type=str, metavar='', required=True,
    help='Specify the plot dir'
)
args = parser.parse_args()
csv_file = args.input
out_dir = args.output
if not os.path.exists(out_dir):
	os.makedirs(out_dir)


dataset = web.get_data_yahoo("SPY", "03/01/2003", "12/01/2015")
print(dataset.head())
dataset.index = dataset.index.tz_localize('UTC')
dataset['Value']=dataset['Adj Close'].pct_change()
returns = pd.Series(dataset['Value'], index=dataset.index)
tearsheet = pf.tears.create_full_tear_sheet(returns)
print(tearsheet)