import matplotlib.pyplot as plt
from utils.misc import get_logger
import os
import logging

plot_log = get_logger(__name__)
plot_log.setLevel(logging.INFO)

def linear_regression_plot(y_test, train_pred, out_path):
    plot_log.info("in linear regression plot")
    plt.plot(y_test.values, color='r', linewidth=1, label='test')
    plt.plot(train_pred, '-.', color="b", linewidth=1, label='pred')
    plt.ylabel('\u0394 Rel')
    plt.legend(loc='upper center', frameon=False)
    plt.title('LinearRegression')
    figname = os.path.join(out_path, "linearRegr_plot"+".png")
    plt.savefig(figname, dpi=200)
    plt.close()

def linear_regression_scatter(y_test, train_pred, out_path):
    plot_log.info("in linear regression scatter")
    plt.scatter(y_test.values, train_pred)
    plt.xlabel('\u0394 Rel')
    plt.ylabel('\u0394 Rel predicted')
    plt.title('\u0394 Rel comparison')
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    figname = os.path.join(out_path, "linearRegr_scatter"+".png")
    plt.savefig(figname, dpi=200)
    plt.close()

def plot_sign(y_test, pred, out_path):
    plot_log.info("in plot sign")
    import numpy as np
    plt.scatter(np.sign(y_test), np.sign(pred))
    plt.xlabel('\u0394 Rel sign')
    plt.ylabel('\u0394 Rel predicted sign')
    plt.title('\u0394 Rel SIGN comparison')
    figname = os.path.join(out_path, "linearRegr_sign"+".png")
    plt.savefig(figname, dpi=200)
    plt.close()
