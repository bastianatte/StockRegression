import matplotlib.pyplot as plt
import pandas as pd
from utils.misc import get_logger
import logging
import os

plotter_log = get_logger(__name__)
plotter_log.setLevel(logging.INFO)

class plotter(object):
    def __init__(self, dataframe, train, test, out_path):
        self.dataframe = dataframe
        self.train = train
        self.test = test
        self.out_path = out_path


    def plot_clop(self):
        """
        Plot "Close - Open" variable
        """
        y = self.dataframe.Clop
        y = pd.Series(y)
        fig = plt.figure()
        y.plot(figsize=(22,8), color='teal')
        plt.title('Clop variable')
        figname = os.path.join(self.out_path, "clop" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_clop_norm(self):
        """
        Plot "(Close - Open)/Open" variable
        """
        y = self.dataframe.Clop_norm
        y = pd.Series(y)
        fig = plt.figure()
        y.plot(figsize=(18,5), color='teal')
        plt.title('\u0394 rel')
        figname = os.path.join(self.out_path, "deltarel" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_close_open(self):
        """
        Plot Close and Open variables
        """
        plotter_log.info("in plot close open")
        dfopen = self.dataframe.Open
        dfclose = self.dataframe.Close
        dfopen = pd.Series(dfopen)
        dfclose = pd.Series(dfclose)
        fig = plt.figure()
        plt.title('"Close" and "Open" price per day')
        dfopen.plot(figsize=(22,8), color='red')
        dfclose.plot(color='cyan')
        plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.out_path, "close_open" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()        

    def plot_train_test(self):
        """
        Plot train and test
        """
        plotter_log.info("in plot train test")
        plt.figure(figsize=(18,5))
        plt.title('Relative delta per day')
        # plt.plot(self.train["Clop_norm"], color='teal')
        # plt.plot(self.test["Clop_norm"], color='orangered')
        plt.plot(self.train.Clop_norm, color='teal')
        plt.plot(self.test.Clop_norm, color='orangered')
        plt.legend(['Train','Test'])
        plt.xlabel('Date')
        plt.ylabel('\u0394 rel')
        figname = os.path.join(self.out_path, "train_test_deltarel" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def check_stationarity(self, lags_plots=48, figsize=(22,8)):
        """
        Function to plot the graph and show the statistic test result.
        """
        import statsmodels.api as sm
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        import seaborn as sns
        from math import sqrt
        import matplotlib.pyplot as plt
        "Use Series as parameter"

        # y = self.train["High"].diff().dropna()
        y = self.train.Clop_norm
        y = pd.Series(y)
        fig = plt.figure()
        ################
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((3, 3), (1, 0))
        ax3 = plt.subplot2grid((3, 3), (1, 1))
        ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        ################
        y.plot(ax=ax1, figsize=figsize, color='teal')
        ax1.set_title('\u0394 Rel')
        plot_acf(y, lags=lags_plots, zero=False, ax=ax2, color='teal');
        plot_pacf(y, lags=lags_plots, zero=False, ax=ax3, method='ols', color='teal');
        sns.distplot(y, bins=int(sqrt(len(y))), ax=ax4, color='teal')
        ax4.set_title('Price Distribution')
        plt.tight_layout()
        ################
        figname = os.path.join(self.out_path, "stationarity_close_norm" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        ################
        print('Dickey-Fuller test results:')
        adfinput = adfuller(y)
        adftest = pd.Series(adfinput[0:4], index=['Statistical Test','P-Value','Used Lags','Observations Number'])
        adftest = round(adftest,4)
        ################
        for key, value in adfinput[4].items():
            adftest["Critical Values (%s)"%key] = value.round(4)
            
        print(adftest)