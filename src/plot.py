import sys
import matplotlib
import matplotlib as mpl
import pylab
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import openpyxl
import math

class Plot():

    def __init__(self, app_list, result_by_app, file_name):
        self.app_list = app_list
        self.result_by_app = result_by_app
        self.width = 0.3
        self.ind = np.arange(len(result_by_app[0]))
        self.num_of_result = len(result_by_app[0])
        self.file_name = file_name

    def make_figure(self):
        self.fig, self.ax = plt.subplots()
        latency_list = []
        app_name_list = []
        color_list = ['black', 'white', 'grey', 'darkred'] # FIXME: assume maximum 4 networks
        for idx, latency in enumerate(self.result_by_app):
            latency_list.append(self.ax.bar(self.ind + self.width * idx, latency, self.width, color=color_list[idx], ec='k'))
            app_name_list.append(self.app_list[idx].name)
        app_name_tuple = tuple(app_name_list)
        latency_tuple = tuple(latency_list)

        self.ax.set_xticks(self.ind + self.width)
        self.ax.set_xticklabels([])
        self.ax.set_xlabel("Solutions", fontsize=15)
        self.ax.set_ylabel("Response Time ($\mu$s)", fontsize=15)
        self.ax.legend(latency_tuple,\
                app_name_tuple, loc='best',labelspacing=0.2)
        if self.num_of_result == 1:
            self.fig.set_size_inches(5, 5)
            self.ax.set_xlim((-1, 1))
        else:
            self.fig.set_size_inches(20, 5)
            self.ax.set_xlim((-1, self.num_of_result))

        self.fig.savefig(self.file_name + '.pdf', bbox_inches="tight", pad_inches=0)

