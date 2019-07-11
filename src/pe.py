from enum import Enum
import itertools
import config

class PEType(Enum):
    CPU = 0
    GPU = 1
    NPU = 2

class PE(object):
    newid = itertools.count().next
    def __init__(self, name):
        self.name = name
        self.idx = PE.newid() # TODO do it really need?
        self.transfer_time = {}

        if 'cpu' in name:
            self.type =  PEType.CPU
            self.cpu_core_num = int(config.cpu_config[self.idx])
            self.energy = 3.5
        elif 'gpu' in name:
            self.type =  PEType.GPU
            self.energy = 4.0
        elif 'npu' in name:
            self.type =  PEType.NPU
            self.energy = 2.0

    @staticmethod
    def init_apps_pe_by_mapping(app_list, mapping, pe_list):
        for app in app_list:
            for layer in app.layer_list:
                pe_idx = mapping[layer.get_index()]
                layer.set_pe(pe_list[pe_idx])

    def get_idx(self):
        return self.idx

    def get_type(self):
        return self.type

    def get_cpu_core_num(self):
        return self.cpu_core_num

    def get_energy(self):
        return self.energy
