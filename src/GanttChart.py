import matplotlib
matplotlib.use('Agg')

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


class GanttChart:
    file_name = None
    ylabels = None
    task_list = None
    fig = None
    ax = None

    def __init__(self, file_name):
        self.file_name = file_name 
        self.task_list = []

    def __init__(self, file_name, processor_list):
        self.file_name = file_name 
        self._set_ylabels(processor_list)

    def _set_ylabels(self, processor_list):
        self.ylabels = []
        for processor_name in processor_list:
            self.ylabels.append(processor_name)
        self.task_list = []

    def add_task(self, result):
        # result_list has tuple (layer_name, processor_name, start_time, end_time, mem_transition_time)
        self.task_list.append([result[0], result[1], result[2], result[3]])
        if result[4] != 0:
            self.task_list.append(['overhead', result[1], result[3], result[3] + result[4]])

    def add_tasks_all(self, result_list):
        # result_list has tuple (layer_name, processor_name, start_time, end_time, mem_transition_time)
        for key in result_list:
            self.add_task(key)
            # save as [layer_name or overhead, processor_name, start_time, end_time]

    def get_color_name(self, layer_name):
        if layer_name.find("data") >= 0 or layer_name.find("front") >= 0:
            return 'grey'
        elif layer_name.find("conv") >= 0 or layer_name.find("c") >= 0: 
            return 'gold'
        elif layer_name.find("pool") >= 0 or layer_name.find("p") >= 0:
            return 'lavender'
        elif layer_name.find("ip") >= 0:
            return 'green'
        else:
            return 'lightblue'

        # XXX Jinwoo START ------------------------------
        # classify apps by color (upto 4 apps supported)
        # if layer_name.find("app_0") >= 0:
        #     return 'gold'
        # if layer_name.find("app_1") >= 0:
        #     return 'green'
        # if layer_name.find("app_2") >= 0:
        #     return 'lightblue'
        # if layer_name.find("app_3") >= 0:
        #     return 'grey'
        # XXX Jinwoo END --------------------------------
        
    def draw_gantt_chart(self):
        num_ylabels = len(self.ylabels)
        pos = np.arange(0.5,num_ylabels *0.5 + 0.5,0.5)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # bar_container_dict = {}
        # bar_start_time_dict = {}
        for each_task in self.task_list:
            if each_task[0] == 'overhead':
                if each_task[3] - each_task[2] != 0:
                    # self.ax.text(each_task[2], self.ylabels.index(each_task[1]) * 0.5 + 0.5, 'middle', horizontalzlignment='center', verticalalignment='center',rotation='vertical',transform=self.ax.transAxes)
                    self.ax.barh(y=(self.ylabels.index(each_task[1]) * 0.5) + 0.5, width=(each_task[3]-each_task[2]), left=each_task[2], height=0.3, align = 'center', color='red', alpha=0.8, gid=each_task[1])
            else:
                for each_processor in self.ylabels:
                    if each_task[1] == each_processor:
                        color_name = self.get_color_name(each_task[0])
                        # self.ax.text(0.95 * (each_task[3] - each_task[2]), 1.5)
                        # self.ax.text(each_task[2], self.ylabels.index(each_processor) * 0.5 + 0.5, each_task[0], horizontalalignment='center', verticalalignment='center',rotation='vertical',transform=self.ax.transAxes)
                        bar = self.ax.barh(y=(self.ylabels.index(each_processor) * 0.5) + 0.5, width=(each_task[3]-each_task[2]), left=each_task[2], height=0.3, align = 'center', color=color_name, edgecolor='black', alpha=0.8, gid=each_task[1])
                        # bar_container_dict[each_task[0]] = bar[0]
                        # bar_start_time_dict[each_task[0]] = each_task[2]
                        if each_task[0] != 'overhead':
                            rect = bar[0]
                            start_time = each_task[2]
                            self.ax.text(start_time + 0.99 * rect.get_width(), rect.get_y() + rect.get_height() / 2.0, each_task[0], ha='right', va='center', rotation='vertical', fontsize = 4)
                        # self.ax.text(0.95 * rect.width, rect.y + rect.height / 2.0, each_task[0], ha='right', va='center')
                        break
        # for key in bar_container_dict:
        #     if key != 'overhead':
        #         rect = bar_container_dict[key]
        #         start_time = bar_start_time_dict[key]
        #         self.ax.text(start_time + 0.99 * rect.get_width(), rect.get_y() + rect.get_height() / 2.0, key, ha='right', va='center', rotation='vertical', fontsize = 4)

        lacsy, labelsy = plt.yticks(pos, self.ylabels)
        plt.setp(labelsy, fontsize = 10)

        # self.ax.axis('tight')
        self.ax.set_ylim(ymin = -0.1, ymax = num_ylabels * 0.5 + 0.5)
        self.ax.grid(color = 'g', linestyle = ':')

        # self.ax.xaxis.set_major_locator(ticker.MultipleLocator(1.00))
        self.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        # self.ax.text(0.0, 0.1, "ScalarFormatter()", fontsize=10, transform=self.ax.transAxes)
        labelsx = self.ax.get_xticklabels()
        plt.setp(labelsx, rotation=30, fontsize=10)

        font = font_manager.FontProperties(size='small')
        self.ax.legend(loc=1, prop=font)

        self.ax.invert_yaxis()
        # plt.show()
        plt.savefig(self.file_name, dpi=250)

