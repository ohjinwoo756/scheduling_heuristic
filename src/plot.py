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
        self.width = 0.2
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
        plt.clf()

        # latency_squeezenet_mobility = [18891.51, 19088.35, 19532.9, 19679.57, 19679.57, 19679.57, 19691.57]
        # latency_mobilenet_v1_mobility = [18389.94, 16920.03, 16806.91, 16329.94, 16329.94, 16342.04, 16329.94]
        # latency_mobilenet_v2_mobility = [33440.83, 32817.97, 29863.47, 31637.79, 30004.42, 35599.07, 35599.07]
        # energy_mobility = [73504, 93041, 95171, 98941, 99436, 98940, 98940]
        # 
        # fig, ax = plt.subplots()
        # ax2 = ax.twinx()
        # ind = np.arange(len(latency_squeezenet_mobility))
        # latency_sq  = ax.bar(ind, latency_squeezenet_mobility,   self.width, color='w', ec='k')
        # latency_mn1 = ax.bar(ind + self.width, latency_mobilenet_v1_mobility, self.width, color='w', hatch='\\\\\\\\\\', ec='k')
        # latency_mn2 = ax.bar(ind + self.width * 2, latency_mobilenet_v2_mobility, self.width, color='grey', ec='k')
        # energy = ax2.plot(ind, energy_mobility, 'k', color='darkred')
        # ax.set_xticks(ind + self.width)
        # ax.set_xticklabels([])
        # ax.set_xlim((-0.5,7))
        # ax.set_ylim((10000, 37000))
        # ax2.set_xlim((-0.5,7))
        # ax2.set_ylim((70000, 100000))
        # ax.set_xlabel("Solutions", fontsize=12)
        # ax.set_ylabel("Response Time ($\mu$s)", fontsize=14)
        # ax2.set_ylabel("Energy consumption (W x $\mu$s)", fontsize=10)
        # ax.legend((latency_sq[0], latency_mn1[0], latency_mn2[0], energy[0]),\
        #         ('SqueezeNet', 'MobileNet v1', 'MobileNet v2', 'Energy consumption'), loc='best',labelspacing=0.1)
        # fig.set_size_inches(7,2.3)
        # fig.savefig('iccd19_3app_schedule_pareto_result_with_energy.pdf', bbox_inches="tight", pad_inches=0)
        # plt.clf()



	# ga_squeezenet_40 = [18395.64, 16335.11, 15677.98, 15394.46, 14015.38] # best: 1111
	# ga_squeezenet_50 = [17203.93, 15517.95, 15510.57, 14961.84, 12825.55] # best: 1111
	# ga_squeezenet_60 = [16331.84, 15338.32, 15355.72, 14034.93, 11721.21] # best: 1111
	# ga_squeezenet_70 = [15797.36, 14500.27, 14483.65, 12895.11, 11221.44] # best: 1111
	# ga_squeezenet_100 = [15797.36, 12971.99, 12933.18, 12527.04, 10651.69] # best: 1111
	# 
	# ga_mobilenet_v1_40 = [23228.97, 21233.99, 21108.41, 19863.96, 19051.91] # best: 1111
	# ga_mobilenet_v1_50 = [21964.55, 20217.49, 20612.79, 18792.01, 17664.67] # best: 1111
	# ga_mobilenet_v1_60 = [21339.32, 19201.57, 19512.32, 17741.86, 16676.01] # best: 1111
	# ga_mobilenet_v1_70 = [20413.31, 18105.85, 18879.86, 16912.93, 15451.17] # best: 1111
	# ga_mobilenet_v1_100 = [19494.68, 17020.07, 17563.57, 15713.15, 14458.99] # best: 1111
	# 
	# ga_mobilenet_v2_40 = [28413.46, 28873.47, 30496.93, 28570.92, 27486.78] # best: 1111
	# ga_mobilenet_v2_50 = [26784.34, 27955.60, 28887.55, 26929.15, 25787.42] # best: 1111
	# ga_mobilenet_v2_60 = [25424.24, 27102.00, 28118.77, 26461.43, 24344.74] # best: 1111
	# ga_mobilenet_v2_70 = [23719.04, 25890.51, 27226.11, 25837.80, 22737.93] # best: 1111
	# ga_mobilenet_v2_100 = [22271.50, 24387.51, 25860.30, 24267.67, 20991.23] # best: 1111
	# 
	# ga_densenet_40 = [39074.25, 38324.67, 37821.08, 35291.63, 34441.41] # best: 1111
	# ga_densenet_50 = [38741.40, 37162.03, 35826.39, 34187.86, 33341.97] # best: 1111
	# ga_densenet_60 = [36387.21, 34667.23, 34669.77, 33423.20, 33341.97] # best: 1111
	# ga_densenet_70 = [35470.71, 34016.38, 34510.50, 33423.20, 33341.97] # best: 1111
	# ga_densenet_100 = [35470.71, 34016.38, 34669.77, 33423.20, 33341.97] # best: 1111
	# 
	# fig, ax = plt.subplots()
	# width = 0.15
	# ind = np.arange(5)
	# ga100 = ax.bar(ind, ga_squeezenet_100, width, color='w', ec='k')
	# ga70 = ax.bar(ind+1*width, ga_squeezenet_70, width, color='darkgrey', ec='k')
	# ga60 = ax.bar(ind+2*width, ga_squeezenet_60, width, color='w', hatch='\\\\\\\\\\', ec='k')
	# ga50 = ax.bar(ind+3*width, ga_squeezenet_50, width, color='grey', ec='k')
	# ga40 = ax.bar(ind+4*width, ga_squeezenet_40, width, color='w', hatch='.....', ec='k')
	# ax.set_xticks(ind + width * 2)
	# ax.set_xticklabels(('4', '22', '31', '211', '1111'))
	# ax.legend((ga100[0], ga70[0], ga60[0], ga50[0], ga40[0]),\
	#         ('100%', '70%', '60%', '50%', '40%'), loc='best', ncol=2, labelspacing=0.3, prop={'size':5})
	# ax.set_ylim((10000,25000))
	# plt.ylabel("1 / Throughput ($\mu$s)", fontsize = 7)
	# # ax.autoscale_view()
	# for i in range(0, 9):
	#     ax.yaxis.get_major_ticks()[i].label.set_fontsize(8)
	# for i in range(0, 5):
	#     ax.xaxis.get_major_ticks()[i].label.set_fontsize(8)
	# fig.set_size_inches(3, 1)
	# fig.savefig('iccd19_1app_squeezenet_GA_throughput.pdf', bbox_inches="tight", pad_inches=0)
        # plt.clf()
	# 
	# fig, ax = plt.subplots()
	# width = 0.15
	# ind = np.arange(5)
	# ilp100 = ax.bar(ind, ga_mobilenet_v1_100, width, color='w', ec='k')
	# ilp70 = ax.bar(ind+1*width, ga_mobilenet_v1_70, width, color='darkgrey', ec='k')
	# ilp60 = ax.bar(ind+2*width, ga_mobilenet_v1_60, width, color='w', hatch='\\\\\\\\\\', ec='k')
	# ilp50 = ax.bar(ind+3*width, ga_mobilenet_v1_50, width, color='grey', ec='k')
	# ilp40 = ax.bar(ind+4*width, ga_mobilenet_v1_40, width, color='w', hatch='.....', ec='k')
	# ax.set_xticks(ind + width * 2)
	# ax.set_xticklabels(('4', '22', '31', '211', '1111'))
	# ax.legend((ga100[0], ga70[0], ga60[0], ga50[0], ga40[0]),\
	#         ('100%', '70%', '60%', '50%', '40%'), loc='best', ncol=2, labelspacing=0.3, prop={'size':5})
	# plt.ylabel("1 / Throughput ($\mu$s)", fontsize = 7)
	# ax.set_ylim((10000,30000))
	# # ax.autoscale_view()
	# for i in range(0, 9):
	#     ax.yaxis.get_major_ticks()[i].label.set_fontsize(8)
	# for i in range(0, 5):
	#     ax.xaxis.get_major_ticks()[i].label.set_fontsize(8)
	# fig.set_size_inches(3,1)
	# fig.savefig('iccd19_1app_mobilenet_v1_GA_throughput.pdf', bbox_inches="tight", pad_inches=0)
        # plt.clf()
	# 
	# fig, ax = plt.subplots()
	# width = 0.15
	# ind = np.arange(5)
	# ilp100 = ax.bar(ind, ga_mobilenet_v2_100, width, color='w', ec='k')
	# ilp70 = ax.bar(ind+1*width, ga_mobilenet_v2_70, width, color='darkgrey', ec='k')
	# ilp60 = ax.bar(ind+2*width, ga_mobilenet_v2_60, width, color='w', hatch='\\\\\\\\\\', ec='k')
	# ilp50 = ax.bar(ind+3*width, ga_mobilenet_v2_50, width, color='grey', ec='k')
	# ilp40 = ax.bar(ind+4*width, ga_mobilenet_v2_40, width, color='w', hatch='.....', ec='k')
	# ax.set_xticks(ind + width * 2)
	# ax.set_xticklabels(('4', '22', '31', '211', '1111'))
	# ax.legend((ga100[0], ga70[0], ga60[0], ga50[0], ga40[0]),\
	#         ('100%', '70%', '60%', '50%', '40%'), loc='best', ncol=2, labelspacing=0.3, prop={'size':5})
	# plt.ylabel("1 / Throughput ($\mu$s)", fontsize = 7)
	# ax.set_ylim((20000,40000))
	# # ax.autoscale_view()
	# for i in range(0, 9):
	#     ax.yaxis.get_major_ticks()[i].label.set_fontsize(8)
	# for i in range(0, 5):
	#     ax.xaxis.get_major_ticks()[i].label.set_fontsize(8)
	# fig.set_size_inches(3,1)
	# fig.savefig('iccd19_1app_mobilenet_v2_GA_throughput.pdf', bbox_inches="tight", pad_inches=0)
        # plt.clf()
	# 
	# fig, ax = plt.subplots()
	# width = 0.15
	# ind = np.arange(5)
	# ilp100 = ax.bar(ind, ga_densenet_100, width, color='w', ec='k')
	# ilp70 = ax.bar(ind+1*width, ga_densenet_70, width, color='darkgrey', ec='k')
	# ilp60 = ax.bar(ind+2*width, ga_densenet_60, width, color='w', hatch='\\\\\\\\\\', ec='k')
	# ilp50 = ax.bar(ind+3*width, ga_densenet_50, width, color='grey', ec='k')
	# ilp40 = ax.bar(ind+4*width, ga_densenet_40, width, color='w', hatch='.....', ec='k')
	# ax.set_xticks(ind + width * 2)
	# ax.set_xticklabels(('4', '22', '31', '211', '1111'))
	# ax.legend((ga100[0], ga70[0], ga60[0], ga50[0], ga40[0]),\
	#         ('100%', '70%', '60%', '50%', '40%'), loc='best', ncol=2, labelspacing=0.3, prop={'size':5})
	# plt.xlabel("CPU configuration", fontsize=8)
	# plt.ylabel("1 / Throughput ($\mu$s)", fontsize = 7)
	# ax.set_ylim((30000,50000))
	# # ax.autoscale_view()
	# for i in range(0, 9):
	#     ax.yaxis.get_major_ticks()[i].label.set_fontsize(8)
	# for i in range(0, 5):
	#     ax.xaxis.get_major_ticks()[i].label.set_fontsize(8)
	# fig.set_size_inches(3,1)
	# fig.savefig('iccd19_1app_densenet_GA_throughput.pdf', bbox_inches="tight", pad_inches=0)
        # plt.clf()



	# hikey_ga_squeezenet_40 = [15809.94, 14508.37, 14500.12, 13906.49, 13572.38]
	# hikey_ga_squeezenet_50 = [14991.26, 13890.89, 14014.38, 13246.74, 12811.02]
	# hikey_ga_squeezenet_60 = [14301.72, 13336.13, 13513.97, 12698.85, 12187.62]
	# hikey_ga_squeezenet_70 = [13836.99, 12774.87, 13221.14, 12187.64, 11802.23]
	# hikey_ga_squeezenet_100 = [12961.71, 11890.47, 12352.66, 11545.63, 10870.65]
	# 
	# hikey_ga_mobilenet_v1_40 = [23170.97, 21848.04, 22210.82, 21150.10, 20676.33]
	# hikey_ga_mobilenet_v1_50 = [22425.59, 21032.77, 21553.22, 20343.69, 19836.58]
	# hikey_ga_mobilenet_v1_60 = [21561.92, 20110.75, 21054.88, 19562.84, 18939.50]
	# hikey_ga_mobilenet_v1_70 = [20882.23, 19523.10, 20469.58, 18746.25, 18182.64]
	# hikey_ga_mobilenet_v1_100 = [19264.46, 17935.80, 19227.01, 17482.87, 16970.32]
	# 
	# hikey_ga_mobilenet_v2_40 = [31276.88, 29076.93, 28926.80, 27631.43, 26853.88]
	# hikey_ga_mobilenet_v2_50 = [29981.66, 27672.38, 27945.36, 26403.34, 25404.90]
	# hikey_ga_mobilenet_v2_60 = [28873.28, 26537.34, 27072.72, 25326.30, 24285.84]
	# hikey_ga_mobilenet_v2_70 = [27793.88, 25615.72, 26210.26, 24380.74, 23372.91]
	# hikey_ga_mobilenet_v2_100 = [25654.06, 23355.45, 24312.69, 22172.03, 21677.32]
	# 
	# hikey_ga_densenet_40 = [40744.95, 38820.48, 39761.54, 38547.87, 37217.44]
	# hikey_ga_densenet_50 = [39140.45, 37625.5, 38936.94, 37157.78, 36303.41]
	# hikey_ga_densenet_60 = [37966.6, 36567.8, 37941.06, 35920.27, 35181.93]
	# hikey_ga_densenet_70 = [37049.52, 35368.15, 37102.77, 35136.5, 34484.06]
	# hikey_ga_densenet_100 = [34649.83, 33913.18, 35654.93, 33636.55, 33757.04]
	# 
	# fig, ax = plt.subplots()
	# width = 0.15
	# ind = np.arange(5)
	# ga100 = ax.bar(ind, hikey_ga_squeezenet_100, width, color='w', ec='k')
	# ga70 = ax.bar(ind+1*width, hikey_ga_squeezenet_70, width, color='darkgrey', ec='k')
	# ga60 = ax.bar(ind+2*width, hikey_ga_squeezenet_60, width, color='w', hatch='\\\\\\\\\\', ec='k')
	# ga50 = ax.bar(ind+3*width, hikey_ga_squeezenet_50, width, color='grey', ec='k')
	# ga40 = ax.bar(ind+4*width, hikey_ga_squeezenet_40, width, color='w', hatch='.....', ec='k')
	# ax.set_xticks(ind + width * 2)
	# ax.set_xticklabels(('4', '22', '31', '211', '1111'))
	# ax.legend((ga100[0], ga70[0], ga60[0], ga50[0], ga40[0]),\
	#         ('100%', '70%', '60%', '50%', '40%'), loc='best', ncol=2, labelspacing=0.3, prop={'size':5})
	# ax.set_ylim((10000,25000))
	# # ax.autoscale_view()
	# # font
	# for i in range(0, 9):
	#     ax.yaxis.get_major_ticks()[i].label.set_fontsize(8)
	# for i in range(0, 5):
	#     ax.xaxis.get_major_ticks()[i].label.set_fontsize(8)
	# fig.set_size_inches(3,1)
	# fig.savefig('iccd19_1app_hikey_squeezenet_GA_throughput.pdf', bbox_inches="tight", pad_inches=0)
        # plt.clf()
	# 
	# fig, ax = plt.subplots()
	# width = 0.15
	# ind = np.arange(5)
	# ga100 = ax.bar(ind, hikey_ga_mobilenet_v1_100, width, color='w', ec='k')
	# ga70 = ax.bar(ind+1*width, hikey_ga_mobilenet_v1_70, width, color='darkgrey', ec='k')
	# ga60 = ax.bar(ind+2*width, hikey_ga_mobilenet_v1_60, width, color='w', hatch='\\\\\\\\\\', ec='k')
	# ga50 = ax.bar(ind+3*width, hikey_ga_mobilenet_v1_50, width, color='grey', ec='k')
	# ga40 = ax.bar(ind+4*width, hikey_ga_mobilenet_v1_40, width, color='w', hatch='.....', ec='k')
	# ax.set_xticks(ind + width * 2)
	# ax.set_xticklabels(('4', '22', '31', '211', '1111'))
	# ax.legend((ga100[0], ga70[0], ga60[0], ga50[0], ga40[0]),\
	#         ('100%', '70%', '60%', '50%', '40%'), loc='best', ncol=2, labelspacing=0.3, prop={'size':5})
	# ax.set_ylim((16000,30000))
	# # ax.autoscale_view()
	# # font
	# for i in range(0, 7):
	#     ax.yaxis.get_major_ticks()[i].label.set_fontsize(8)
	# for i in range(0, 5):
	#     ax.xaxis.get_major_ticks()[i].label.set_fontsize(8)
	# fig.set_size_inches(3,1)
	# fig.savefig('iccd19_1app_hikey_mobilenet_v1_GA_throughput.pdf', bbox_inches="tight", pad_inches=0)
        # plt.clf()
	# 
	# fig, ax = plt.subplots()
	# width = 0.15
	# ind = np.arange(5)
	# ga100 = ax.bar(ind, hikey_ga_mobilenet_v2_100, width, color='w', ec='k')
	# ga70 = ax.bar(ind+1*width, hikey_ga_mobilenet_v2_70, width, color='darkgrey', ec='k')
	# ga60 = ax.bar(ind+2*width, hikey_ga_mobilenet_v2_60, width, color='w', hatch='\\\\\\\\\\', ec='k')
	# ga50 = ax.bar(ind+3*width, hikey_ga_mobilenet_v2_50, width, color='grey', ec='k')
	# ga40 = ax.bar(ind+4*width, hikey_ga_mobilenet_v2_40, width, color='w', hatch='.....', ec='k')
	# ax.set_xticks(ind + width * 2)
	# ax.set_xticklabels(('4', '22', '31', '211', '1111'))
	# ax.legend((ga100[0], ga70[0], ga60[0], ga50[0], ga40[0]),\
	#         ('100%', '70%', '60%', '50%', '40%'), loc='best', ncol=2, labelspacing=0.3, prop={'size':5})
	# ax.set_ylim((20000,40000))
	# # ax.autoscale_view()
	# # font
	# for i in range(0, 7):
	#     ax.yaxis.get_major_ticks()[i].label.set_fontsize(8)
	# for i in range(0, 5):
	#     ax.xaxis.get_major_ticks()[i].label.set_fontsize(8)
	# fig.set_size_inches(3,1)
	# fig.savefig('iccd19_1app_hikey_mobilenet_v2_GA_throughput.pdf', bbox_inches="tight", pad_inches=0)
        # plt.clf()
	# 
	# fig, ax = plt.subplots()
	# width = 0.15
	# ind = np.arange(5)
	# ga100 = ax.bar(ind, hikey_ga_densenet_100, width, color='w', ec='k')
	# ga70 = ax.bar(ind+1*width, hikey_ga_densenet_70, width, color='darkgrey', ec='k')
	# ga60 = ax.bar(ind+2*width, hikey_ga_densenet_60, width, color='w', hatch='\\\\\\\\\\', ec='k')
	# ga50 = ax.bar(ind+3*width, hikey_ga_densenet_50, width, color='grey', ec='k')
	# ga40 = ax.bar(ind+4*width, hikey_ga_densenet_40, width, color='w', hatch='.....', ec='k')
	# ax.set_xticks(ind + width * 2)
	# ax.set_xticklabels(('4', '22', '31', '211', '1111'))
	# ax.legend((ga100[0], ga70[0], ga60[0], ga50[0], ga40[0]),\
	#         ('100%', '70%', '60%', '50%', '40%'), loc='best', ncol=2, labelspacing=0.3, prop={'size':5})
	# plt.xlabel("CPU configuration", fontsize=8)
	# plt.annotate("", xytext=(3, 35200), xy=(3,33800), arrowprops={'facecolor':'crimson', 'arrowstyle':'simple', 'ec':'crimson'})
	# ax.set_ylim((32000,50000))
	# # ax.autoscale_view()
	# # font
	# for i in range(0, 6):
	#     ax.yaxis.get_major_ticks()[i].label.set_fontsize(8)
	# for i in range(0, 5):
	#     ax.xaxis.get_major_ticks()[i].label.set_fontsize(8)
	# fig.set_size_inches(3,1)
	# fig.savefig('iccd19_1app_hikey_densenet_GA_throughput.pdf', bbox_inches="tight", pad_inches=0)
        # plt.clf()



	# energy_squeezenet = [61566, 61703, 61796, 61932, 61938, 62145, 62239, 62683, 63667, 65160, 65160]
	# throughput_squeezenet = [19812.87, 17668.41, 16897.91, 15975.67, 15960.63, 15832.22, 13965.16, 12813.02, 11974.86, 11877.93, 11721.21]
	# energy_mobilenet_v2 = [133395, 133399, 133525, 133572, 133576, 133702, 133923, 134103, 134107, 134234, 134406, 134410, 134438, 134490, 134494, 134615, 134621, 134667, 134671, 134798, 135018, 135446, 135491, 135557, 135601, 136030, 136136, 136140, 137372, 137391, 137501, 138997, 139026, 139111]
	# throughput_mobilenet_v2 = [38514.22, 37782.42, 37498.88, 37464.69, 36719.31, 36364.64, 35806.77, 35605.1, 35114.19, 33927.8, 33647.94, 33281.46, 33126.1, 32990.35, 32260.5, 32084.12, 31973.14, 30916.36, 30391.12, 29588.23, 28861.46, 28444.98, 28397.09, 27696.73, 27338.76, 27165.58, 26544.87, 25869.36, 25844.08, 25548.79, 24991.13, 24774.57, 24766.5, 24509.12]
	# 
	# throughput_energy_squeezenet = plt.plot(throughput_squeezenet, energy_squeezenet, 'k-')
	# plt.xlabel("1 / Throughput ($\mu$s)", fontsize=13)
	# plt.ylabel("Energy consumption\n(W x $\mu$s)", fontsize=11)
	# plt.xticks([12000, 15000, 20000], fontsize=13)
	# plt.yticks(fontsize=13)
	# fig = plt.gcf()
	# fig.set_size_inches(5,2)
	# fig.savefig('iccd19_throughput_energy_squeezenet.pdf', bbox_inches="tight", pad_inches=0)
        # plt.clf()
	# 
	# throughput_energy_mobilenet_v2 = plt.plot(throughput_mobilenet_v2, energy_mobilenet_v2, 'k-')
	# plt.xlabel("1 / Throughput ($\mu$s)", fontsize=13)
	# plt.xticks([25000, 30000, 35000, 38000], fontsize=13)
	# plt.yticks(fontsize=13)
	# fig = plt.gcf()
	# fig.set_size_inches(5,2)
	# fig.savefig('iccd19_throughput_energy_mobilenet_v2.pdf', bbox_inches="tight", pad_inches=0)
        # plt.clf()
