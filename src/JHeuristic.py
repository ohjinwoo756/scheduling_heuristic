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
        self.fitness = Fitness(app_list, pe_list) # compute mapping's fitness

        self.optimistic_cost_table = list()
        self.optimistic_cost_hash = dict() # for speed up

        self.processor_available_time_list = [0] * self.num_pe


    def do_schedule(self):
        self.make_optimistic_cost_table()
        self.peft_algorithm()

        return self.get_mappings()


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
        # put entry layer per application as initial task
        for app_idx in range(0, len(self.app_list)):
            ready_list.append(self.app_list[app_idx].layer_list[0])

        target_layers = list()
        layers_rank_oct = dict() # layers' rank oct
        for app in self.app_list:
            for layer in app.layer_list: # initial target layers : ALL layers
                target_layers.append(layer)
                layers_rank_oct[layer] = layer.get_rank_oct()

        ready_list = sorted(ready_list, key=layers_rank_oct.__getitem__, reverse=True) # initial sorting
        # XXX: entry layer per application removed for next function call 'update_layer_ready_list'
        for app_idx in range(0, len(self.app_list)):
            layer = self.app_list[app_idx].layer_list[0]
            target_layers.remove(layer)

        # print self.processor_available_time_list
        # print "  "
        while ready_list != []: # until ready_list is NOT Empty
            # print "ready_list: ", ready_list
            highest_prio_layer = ready_list[0] # the first one
            min_oeft_processor = None
            min_oeft = float('inf')

            if highest_prio_layer.is_start_node or highest_prio_layer.is_end_node: # if frontend or backend
                target_pe_list = self.pe_list[:len(config.cpu_config)] # XXX: if node is frontend or backend, only cpu processor can be mapped.
            else:
                target_pe_list = self.pe_list # XXX: other layers can be mapped to any processor.

            for processor in target_pe_list:
                oeft = self.compute_oeft(highest_prio_layer, processor)
                if oeft < min_oeft: # XXX: processor selection phase
                    min_oeft_processor = processor
                    min_oeft = oeft

            # update actual layer's pe info
            highest_prio_layer.set_pe(min_oeft_processor)
            # print "highest_prio_layer: ", highest_prio_layer.name
            # print "min_oeft_processor: ", min_oeft_processor.name
            # print "BEFORE | ", self.processor_available_time_list

            # XXX: no need for temporal assignment, because it was already mapped in real.
            # update processor_available_time_list
            in_edges_list = list(highest_prio_layer.app.graph._in_edge[highest_prio_layer])
            if in_edges_list == []: # if there is no preceding edge (entry node)
                self.processor_available_time_list[min_oeft_processor.idx] = \
                        round(highest_prio_layer.time_list[min_oeft_processor.idx], 2)
            else:
                max_pre_finish_and_comm_time = -float('inf')
                for e in in_edges_list:
                    self.assign_temporal_edge_type_between(e.sender, highest_prio_layer) # assign pe
                    # print "sender: ", e.sender, " | receiver: ", e.receiver
                    # print "e.sender.finish_time", e.sender.finish_time
                    # print "e.calc_transition_time: ", e.calc_transition_time()
                    pre_finish_and_comm_time = e.sender.finish_time + \
                            e.calc_transition_time()
                    if pre_finish_and_comm_time > max_pre_finish_and_comm_time:
                        max_pre_finish_and_comm_time = pre_finish_and_comm_time
                self.processor_available_time_list[min_oeft_processor.idx] = \
                        max_pre_finish_and_comm_time + \
                        round(highest_prio_layer.time_list[min_oeft_processor.idx], 2)

            # print "AFTER | ", self.processor_available_time_list

            # update processor_available_time_list to layer's finish time
            highest_prio_layer.finish_time = self.processor_available_time_list[min_oeft_processor.idx]
            # print "highest_prio_layer.finish_time: ", highest_prio_layer.finish_time
            # print "  "

            # update layer's pe_mapped value
            highest_prio_layer.set_pe_mapped(True)

            # update ready list
            ready_list = self.update_layer_ready_list(target_layers, ready_list, layers_rank_oct)


    def compute_oeft(self, layer, processor):
        oct_value = self.optimistic_cost_table[layer.index][processor.idx]
        eft = self.compute_eft(layer, processor)
        return oct_value + eft


    def compute_eft(self, layer, processor):
        est = self.compute_est(layer, processor)
        return est + layer.time_list[processor.idx]


    def compute_est(self, layer, processor):
        in_edges_list = list(layer.app.graph._in_edge[layer])
        if in_edges_list == []: # If there is no preceding edge
            return 0
        else:
            pe_available_time = self.processor_available_time_list[processor.idx]

            precedents = list()
            for e in in_edges_list:
                precedents.append(e.sender)

            max_sum_aft_comm = -float('inf')
            for prior in precedents:
                aft = prior.finish_time

                # XXX: precedents already have assigned processor
                # XXX: processor assigned to layer temporarily
                self.assign_temporal_processor(layer, processor)
                self.assign_temporal_edge_type_between(prior, layer)

                comm_time = prior.app.graph._edge[(prior, layer)].calc_transition_time()
                sum_aft_comm = aft + comm_time
                if sum_aft_comm > max_sum_aft_comm:
                    max_sum_aft_comm = sum_aft_comm

                # XXX: initialize temporal assignment
                self.initialize_temporal_assigned(layer)
                self.initialize_temporal_edge_type_between(prior, layer)

            return max(pe_available_time, max_sum_aft_comm)


    def update_layer_ready_list(self, target_layers, ready_list, layers_rank_oct):
        del ready_list[0] # remove already mapped layer
        runnable_layers_list = self.extract_runnable_layers_from(target_layers)
        for l in runnable_layers_list: # insert runnable layers to ready_list
            ready_list.append(l)
        ready_list = sorted(ready_list, key=layers_rank_oct.__getitem__, reverse=True) # sort

        return ready_list


    def extract_runnable_layers_from(self, target_layers):
        runnable_layers_list = list()
        for layer in target_layers:
            is_runnable = True
            if list(layer.app.graph._in_edge[layer]) != []: # if precedent exists
                for in_edges in list(layer.app.graph._in_edge[layer]):
                    if not in_edges.sender.pe_mapped:
                        is_runnable = False
                        break
                    
            if is_runnable:
                runnable_layers_list.append(layer)

        # update target layers
        for layer in runnable_layers_list:
            target_layers.remove(layer)

        return runnable_layers_list


    def get_mappings(self):
        # XXX: mappings = [[mapping], [mapping], ...] # pareto result possible
        # XXX: But in JHeuristic, there is only one solution
        mappings = [0] * len(self.layer_list)
        idx = 0
        for app in self.app_list:
            for layer in app.layer_list:
                mappings[idx] = layer.pe.get_idx()
                idx = idx + 1

        # debug
        # print [mappings]
        return [mappings]


