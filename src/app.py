from Layer import Layer
import config


class Application:
    def __init__(self, name, graph, layer_list, period, priority, num_prev_concat):
        self.name = name
        self.graph = graph # layer_graph
        self.layer_list = [l for l in layer_list if l.time_list is not None]
        for l in self.layer_list:
            l.set_app(self)

        self.period = period
        self.priority = int(priority)

        self.obj = None
        self.cst = None

        concat_layer_idx = [idx for idx, l in enumerate(layer_list) if l.time_list is None]
        self.num_concat = 0
        tmp = 0
        for i, l in enumerate(layer_list):
            l.set_index(num_prev_concat + self.num_concat)
            if tmp < len(concat_layer_idx) and concat_layer_idx[tmp] <= i:
                tmp = tmp + 1
                self.num_concat = self.num_concat + 1
        del concat_layer_idx
        if config.analyzer == 'rt':
            self.simplified_graph = self.graph.simplify_graph(self.layer_list[0])

    def __repr__(self):
        return repr((self.name, self.priority, self.period))

    def get_num_layer(self):
        return len(self.layer_list)

    def set_obj(self, obj):
        self.obj = obj

    def set_cst(self, cst):
        self.cst = cst

    def get_num_concat(self):
        return self.num_concat

    def get_total_utilization(self, mapping, period):
        total_exec_time = 0.0
        for idx, l in enumerate(self.layer_list):
            pe = mapping[idx]
            if pe < len(config.cpu_config):
                total_exec_time += l.time_list[pe] * int(config.cpu_config[pe])
        util = total_exec_time / period / config.num_pe

        return util

    def get_total_exec_time(self, mapping):
        total_exec_time = 0.0
        for idx, l in enumerate(self.layer_list):
            pe = mapping[idx]
            if pe < len(config.cpu_config):
                total_exec_time += l.time_list[pe] * int(config.cpu_config[pe])

        return total_exec_time

    def get_utilization(self, pe):
        pe_exec_time = 0.0
        for l in self.layer_list:
            if pe == l.pe.get_pe_idx() and pe < len(config.cpu_config):
                pe_exec_time += l.time_list[pe]
        util = pe_exec_time / self.period

        return util

    def get_period(self):
        return self.period

    def get_priority(self):
        return self.priority

    def update_mobility(self, node, mobility):
        self.graph.propagate_mobility(node, mobility)

    def do_init(self):
        self.graph.do_init(self.layer_list)
        self.graph.set_edge_type()

    def check_runnable(self, layer, time):
        return self.graph.check_runnable(layer, time)

    def do_layer(self, layer, pe, time):
        return self.graph.do_layer(layer, pe, time)
