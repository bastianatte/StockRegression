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

    def plot_actual_day_rt(self):
        """
        Plot "(Close - Open)/Open" variable
        """
        y = self.dataframe.actual_day_rt
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
        plotter_log.info("plot close open DONE!")

    def plot_train_test(self):
        """
        Plot train and test
        """
        plotter_log.info("in plot train test")
        plt.figure(figsize=(18,5))
        plt.title('Relative delta per day')
        # plt.plot(self.train["Clop_norm"], color='teal')
        # plt.plot(self.test["Clop_norm"], color='orangered')
        plt.plot(self.train.actual_day_rt, color='teal')
        plt.plot(self.test.actual_day_rt, color='orangered')
        plt.legend(['Train','Test'])
        plt.xlabel('Date')
        plt.ylabel('\u0394 rel')
        figname = os.path.join(self.out_path, "train_test_deltarel" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        plotter_log.info("plot train test DONE")