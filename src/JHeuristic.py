from mapping_function import MapFunc
from fitness import Fitness
from pe import PEType
from collections import defaultdict
import config

class JHeuristic(MapFunc):

    def __init__(self, app_list, pe_list):
        self.app_list = app_list
        self.layer_list = [l for app in app_list for l in app.layer_list]
        self.pe_list = pe_list
        self.num_app = len(app_list)
        self.num_layer = len(self.layer_list)
        self.num_pe = len(pe_list)
        self.fitness = Fitness(app_list, pe_list) # compute mapping's fitness value

        self.optimistic_cost_table = list()
        self.optimistic_cost_hash = dict() # for speed up


    def do_schedule(self):
        self.make_optimistic_cost_table()
        self.peft_algorithm()

        # mappings = [net_1_mappings, net_2_mappings, ...]
        # mappings = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2, 0, 0, 0, 0] # mnv1
        # return [mappings] # XXX: [ [], [], [], ... ]: pareto results
        return [] # no solution


    def make_optimistic_cost_table(self):
        for app in self.app_list:
            for layer in app.layer_list:
                oct_layer_row = list() # represents a row(task) in oct (optimistic cost table)
                for pe in self.pe_list:
                    oct_layer_row.append(self.compute_optimistic_cost(app, layer, pe))

                # compute rank_oct (averaged)
                rank_oct = 0
                for idx in range(0, len(oct_layer_row)):
                    rank_oct = rank_oct + oct_layer_row[idx]
                rank_oct = rank_oct / len(self.pe_list)
                oct_layer_row.append(round(rank_oct, 2))

                self.optimistic_cost_table.append(oct_layer_row) # update table
                layer.set_rank_oct(rank_oct) # update layer's rank_oct info

        # debug
        # self.print_optimistic_cost_table()


    def compute_optimistic_cost(self, app, task, processor):
        if (task.name, processor.name) in self.optimistic_cost_hash: # if already computed
            return self.optimistic_cost_hash[(task.name, processor.name)]

        else: # if computed first time
            if task.is_end_node:
                self.optimistic_cost_hash[(task.name, processor.name)] = 0.0
                return self.optimistic_cost_hash[(task.name, processor.name)]

            else:
                max_value = -float('inf')
                for successor in task.app.graph._graph[task]:
                    min_value = float('inf')
                    for pe in self.pe_list:
                        # XXX: processor assigned to layer temporarily
                        self.assign_temporal_processor(task, processor)
                        self.assign_temporal_processor(successor, pe)
                        self.assign_temporal_edge_type_between(task, successor)

                        value = self.compute_optimistic_cost(app, successor, pe) + \
                                    successor.time_list[pe.idx] + \
                                    app.graph._edge[(task, successor)].calc_transition_time()
                        
                        # XXX: initialize temporal assignment
                        self.initialize_temporal_assigned(task)
                        self.initialize_temporal_assigned(successor)
                        self.initialize_temporal_edge_type_between(task, successor)

                        if value < min_value:
                            min_value = value

                    if min_value > max_value:
                        max_value = min_value

                optimistic_cost = round(max_value, 2) # round from float point 2
                self.optimistic_cost_hash[(task.name, processor.name)] = optimistic_cost # for speed up 
                return optimistic_cost


    def assign_temporal_processor(self, layer, pe):
        layer.set_pe(pe)


    def assign_temporal_edge_type_between(self, task, successor):
        if task.pe == successor.pe:
            pass # no communication overhead between exactly same processors
        else:
            task.app.graph._edge[(task, successor)].transition_flag = True
            if task.pe.get_type() == PEType.CPU and successor.pe.get_type() == PEType.GPU:
                task.app.graph._edge[(task, successor)].cpu2gpu = True
            elif task.pe.get_type() == PEType.GPU and successor.pe.get_type() == PEType.CPU:
                task.app.graph._edge[(task, successor)].gpu2cpu = True
            elif task.pe.get_type() == PEType.CPU and successor.pe.get_type() == PEType.NPU:
                task.app.graph._edge[(task, successor)].cpu2npu = True
            elif task.pe.get_type() == PEType.NPU and successor.pe.get_type() == PEType.CPU:
                task.app.graph._edge[(task, successor)].npu2cpu = True
            elif set([task.pe.get_type(), successor.pe.get_type()]) == set([PEType.GPU, PEType.NPU]):
                task.app.graph._edge[(task, successor)].gpu_npu_connection = True


    def initialize_temporal_assigned(self, layer):
        layer.set_pe(None)


    def initialize_temporal_edge_type_between(self, task, successor):
        task.app.graph._edge[(task, successor)].transition_flag = False 
        task.app.graph._edge[(task, successor)].cpu2gpu = False 
        task.app.graph._edge[(task, successor)].gpu2cpu = False 
        task.app.graph._edge[(task, successor)].cpu2npu = False 
        task.app.graph._edge[(task, successor)].npu2cpu = False 
        task.app.graph._edge[(task, successor)].gpu_npu_connection = False


    def print_optimistic_cost_table(self):
        table_row_idx = 0
        for app in self.app_list:
            for layer in app.layer_list:
                print self.optimistic_cost_table[table_row_idx]
                table_row_idx = table_row_idx + 1


    def peft_algorithm(self):
        ready_list = list()
        ready_list.append(self.layer_list[0]) # put entry layer as initial task

        target_layers = list() # entire target layers
        layers_rank_oct = dict() # layers' rank oct value
        for app in self.app_list:
            for layer in app.layer_list:
                target_layers.append(layer)
                layers_rank_oct[layer] = layer.get_rank_oct()
        # XXX: entry layer removed from entire target layers bacause it was processed.
        target_layers.remove(self.layer_list[0])

        while ready_list != []: # until ready_list is NOT Empty
            highest_prio_layer = ready_list[0] # the first one
            min_oeft_processor = None
            min_oeft_value = float('inf')
            for processor in self.pe_list:
                oeft_value = self.compute_oeft_value(highest_prio_layer, processor)
                if oeft_value < min_oeft_value:
                    min_oeft_processor = processor
                    min_oeft_value = oeft_value

            # XXX: update layer's processor info
            highest_prio_layer.set_pe(min_oeft_processor)
            ready_list = self.update_layer_ready_list(target_layers, ready_list, layers_rank_oct)


    def compute_oeft_value(self, layer, processor):
        oct_value = self.optimistic_cost_table[layer.index][processor.idx]
        eft_value = self.compute_eft_value(layer, processor)
        return oct_value + eft_value


    def compute_eft_value(self, layer, processor):
        est_value = self.compute_est_value(layer, processor)
        return est_value + layer.time_list[processor.idx]


    def compute_est_value(self, layer, processor):
        in_edges_list = list(layer.app.graph._in_edge[layer])
        if in_edges_list == []: # If there is no preceding edge
            return 0
        else:
            pe_available_time = self.compute_pe_available_time(processor)

            precedents = list()
            for e in in_edges_list:
                precedents.append(e.sender)

            max_sum_aft_comm = -float('inf')
            for prior in precedents:
                aft_value = self.compute_aft_value(prior)

                # XXX: precedents already have assigned processor
                # XXX: processor assigned to layer temporarily
                self.assign_temporal_processor(layer, processor)
                self.assign_temporal_edge_type_between(prio, layer)

                comm_time = app.graph._edge[(prior, layer)].calc_transition_time()
                sum_aft_comm = aft_value + comm_time
                if sum_aft_comm > max_sum_aft_comm:
                    max_sum_aft_comm = sum_aft_comm

                # XXX: initialize temporal assignment
                self.initialize_temporal_assigned(layer)
                self.initialize_temporal_edge_type_between(prior, layer)

            return max(pe_available_time, max_sum_aft_comm)


    # TODO: not implemented yet
    def compute_pe_available_time(self, processor):
        return 100


    # TODO: not implemented yet
    def compute_aft_value(self, layer):
        return 100


    def update_layer_ready_list(self, target_layers, ready_list, layers_rank_oct):
        ready_list.remove(ready_list[0]) # remove already-pe-assigned layer
        runnable_layers_list = self.extract_runnable_layers_from(target_layers)
        for l in runnable_layers_list: # insert runnable layers to ready_list
            ready_list.append(l)
        ready_list = sorted(ready_list, key=layers_rank_oct.__getitem__, reverse=True) # sort
        return ready_list


    # TODO: not implemented yet
    def extract_runnable_layers_from(self, target_layers):
        runnable_layers_list = list()
        # 1. extract
        # 2. remove
        return runnable_layers_list
