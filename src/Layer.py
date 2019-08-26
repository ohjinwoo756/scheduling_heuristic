from enum import Enum
import itertools
import config
from pe import PE


class LayerType(Enum):
    NONE = -1
    CONV = 0
    FULLY = 1
    POOL = 2
    CONCAT = 3
    SHORTCUT = 4
    FRONT = 5
    BACK = 6
    MERGED = 7


class Layer(object):
    newid = itertools.count().next
    unique_prio = itertools.count().next
    name = None
    layer_type = LayerType.NONE

    need_in_edge_check = False  # 'False' means this layer is NOT schedulable.
    # 'True' means that need to check whether this layer is schedulable.
    need_out_edge_check = True  # 'False' means this layer is NOT schedulable.
    # 'True' means that need to check whether this layer is schedulable.

    time_list = None
    mem_size = None

    # FIXME
    # too platform specific. need to insert an abstraction layer.
    # may be (class Platform)
    map_time = None
    unmap_time = None
    memcpy_time = None

    size = None
    num_output = None

    pad = None
    stride = None
    kernel_size = None

    pe = None
    rank_oct = None

    def __init__(self, name, layer_type, num_output, in_size=[0, 0], kernel_size=0, pad=0, stride=1, is_start_node=False, is_end_node=False):
        self.unique_index = Layer.newid()
        self.index = self.unique_index
        self.is_start_node = is_start_node
        self.is_end_node = is_end_node
        self.app = None
        self.name = name
        self.layer_type = layer_type

        self.priority = -1
        self.iteration = 0

        # FIXME
        self.mobility = 0

        self.kernel_size = kernel_size
        self.num_output = num_output

        self.pad = pad
        self.stride = stride

        if (stride == 1 and pad == 0) or kernel_size == 0:
            self.size = in_size
        else:
            self.size = self._calc_out_size(in_size, kernel_size, pad, stride)
        self.mem_size = self.size[0] * self.size[1] * num_output

        # FIXME hardcoding..
        # Galaxy S9
        self.map_time = 0.0053 * self.mem_size + 569.25
        self.unmap_time = 0.0024 * self.mem_size + 526.39
        self.memcpy_time = 6 * pow(10, -10) * pow(self.mem_size, 2) + 7 * pow(10, -5) * self.mem_size + 1.6979
        # Hikey 970
        # self.map_time = 1e-05 * self.mem_size + 14.261
        # self.unmap_time = 2e-05 * self.mem_size + 16.952
        # self.memcpy_time = 0.0009 * self.mem_size

        self.offset = -1

        self.need_out_edge_check = True
        if self.is_start_node:
            self.offset = 0
            self.need_in_edge_check = True

        self.start_time = 0
        self.finish_time = None
        self.pe_mapped = False

    def set_offset(self, o):
        self.offset = o

    def set_mobility(self, m):
        self.mobility = m

    def __repr__(self):
        return repr((self.app.name, self.name))

    def set_app(self, app):
        self.app = app

    def get_app(self):
        return self.app

    def get_period(self):
        return self.app.get_period()

    def get_app_index(self):
        return self.app_index

    def set_index(self, num_concat_before):
        self.index -= num_concat_before
        if self.is_start_node:
            config.start_nodes_idx.append(self.index + 1)
        self.app_index = self.index - (config.start_nodes_idx[-1] - 1)
        if self.is_end_node:
            config.end_nodes_idx.append(self.index + 1)

    def do_init(self):
        if self.offset >= self.get_period():
            self.set_offset(self.offset - self.iteration * self.get_period())
        self.iteration = 0
        self.need_out_edge_check = True
        if not self.is_start_node:
            self.need_in_edge_check = False
        else:
            self.need_in_edge_check = True

    def increase_iter(self):
        self.iteration += 1

    def set_pe(self, pe):
        self.pe = pe

    def get_pe(self):
        return self.pe

    def _calc_out_size(self, in_size, kernel_size, pad, stride):
        in_height = (in_size[0] + 2 * pad - kernel_size) / stride + 1
        in_width = (in_size[1] + 2 * pad - kernel_size) / stride + 1
        size = [in_height, in_width]
        return size

    def get_unique_index(self):
        return self.unique_index

    def get_index(self):
        return self.index

    def get_name(self):
        return self.name

    def set_time_list(self, time_list):
        self.time_list = time_list

    def set_increase_prio(self):
        self.priority = Layer.unique_prio()

    def set_rank_oct(self, rank_oct):
        self.rank_oct = rank_oct

    def get_priority(self):
        return self.priority

    def get_app_priority(self):
        return self.app.get_priority()

    def get_num_output(self):
        return self.num_output

    # TODO change function name
    def get_size(self):
        return self.size

    # FIXME
    def get_map_time(self):
        return self.map_time

    def get_unmap_time(self):
        return self.unmap_time

    def get_memcpy_time(self):
        return self.memcpy_time

    def get_mem_size(self):
        return self.mem_size

    def get_rank_oct(self):
        return self.rank_oct

    def set_start_time(self, time):
        self.start_time = time

    def get_start_time(self):
        return self.start_time

    def set_finish_time(self, time):
        self.finish_time = time

    def get_finish_time(self):
        return self.finish_time

    def set_pe_mapped(self, is_mapped):
        self.pe_mapped = is_mapped

    def get_pe_mapped(self):
        return self.pe_mapped

    def __str__(self):
        return '{:>10} ({:>3}) | {:>4} x {:>4} x {:>5} | {}'.format(self.name, str(self.priority), str(self.size[1]), str(self.size[0]), str(self.num_output), str(self.time_list))
