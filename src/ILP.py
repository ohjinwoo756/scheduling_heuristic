import sys
from mapping_function import MapFunc

from ctypes import *

class ILP(MapFunc):
    porting = None
    objectives = None
    options = None

    def __init__(self, layer_struct, pe_list):
        # not used
        self.porting = CDLL('./main_porting.so')
        self.objectives = 1 # 0: Latency, 1: Throughput
        self.layer_list = [l for l in layer_struct[1] if l.time_list is not None] # except reshape(concat)
        self.num_layer = len(self.layer_list)

    def set_options(self, options):
        self.options = options

    def do_schedule(self):
        self.porting.ilp_util(self.options.net, self.options.est, int(self.options.cpu_utilization), 100)
        self.porting.get_mapping.restype = c_int
        self.porting.get_throughput.restype = c_float
        mapping = []

        for i in range(0, self.num_layer):
            mapping.append(self.porting.get_mapping(i))
        print "\n===================================================================================="

        # print("ILP throughput: " + str(self.porting.get_throughput()))
        
        return [mapping]
