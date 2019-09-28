from mapping_function import MapFunc
from fitness import Fitness
from pe import PEType
from collections import defaultdict
from Layer import Layer
import config


class JHeuristic(MapFunc):

    def __init__(self, app_list, pe_list):
        self.app_list = app_list
        self.layer_list = [l for app in app_list for l in app.layer_list]
        self.pe_list = pe_list
        self.num_app = len(app_list)
        self.num_layer = len(self.layer_list)
        self.num_pe = len(pe_list)
        self.fitness = Fitness(app_list, pe_list)

        self.optimistic_cost_table = list()
        self.optimistic_cost_hash = dict() # for speed up

        self.processor_available_time_list = [0] * self.num_pe
        self.mapped_layers_per_pe = [[] for _ in range(self.num_pe)]

        # FIXME: deprecated
        self.current_vacancy_per_pe = [0] * self.num_pe


    def do_schedule(self):
        self.calculate_oct_and_rank_oct()
        self.scheduling_heuristic()
        final_mappings = self.get_mappings()
        # self.print_mapped_layers_on_each_pe()

        return final_mappings


    def calculate_oct_and_rank_oct(self):
        for app in self.app_list:
            for layer in app.layer_list:
                oct_layer_row = list() # a row(a task) in oct
                for pe in self.pe_list:
                    oct_layer_row.append(self.compute_optimistic_cost(app, layer, pe))
                rank_oct = 0
                for idx in range(0, len(oct_layer_row)):
                    rank_oct = rank_oct + oct_layer_row[idx]
                rank_oct = rank_oct / len(self.pe_list)
                oct_layer_row.append(round(rank_oct, 2))

                self.optimistic_cost_table.append(oct_layer_row) # update table
                layer.set_rank_oct(rank_oct)

        # self.print_optimistic_cost_table()


    def compute_optimistic_cost(self, app, task, processor):
        if (task.name, processor.name) in self.optimistic_cost_hash:
            return self.optimistic_cost_hash[(task.name, processor.name)] # if already computed, use hash
        else:
            if task.is_end_node:
                self.optimistic_cost_hash[(task.name, processor.name)] = 0.0 # update hash
                return self.optimistic_cost_hash[(task.name, processor.name)]
            else:
                max_value = -float('inf')
                for successor in task.app.graph._graph[task]:
                    min_value = float('inf')
                    for pe in self.pe_list:
                        # XXX: assign PE temporarily (this is for getting transition time below)
                        self.assign_processor(task, processor)
                        self.assign_processor(successor, pe)
                        self.assign_edge_type_between(task, successor)

                        value = self.compute_optimistic_cost(app, successor, pe) + \
                                    successor.time_list[pe.idx] + \
                                    app.graph._edge[(task, successor)].calc_transition_time()
                        
                        # initialize temporal assignment
                        self.initialize_assigned(task)
                        self.initialize_assigned(successor)
                        self.initialize_edge_type_between(task, successor)

                        if value < min_value:
                            min_value = value

                    if min_value > max_value:
                        max_value = min_value

                optimistic_cost = round(max_value, 2)
                self.optimistic_cost_hash[(task.name, processor.name)] = optimistic_cost # update hash
                return optimistic_cost


    # XXX: PAIR: initialize_assigned()
    def assign_processor(self, layer, pe):
        layer.set_pe(pe)


    # XXX: PAIR: initialize_edge_type_between()
    def assign_edge_type_between(self, task, successor):
        if task.pe == successor.pe:
            pass
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


    def _assign_edge_type_from_priors_to(self, successor):
        in_edges = list(successor.app.graph._in_edge[successor])
        if in_edges != []:
            for e in in_edges:
                prior = e.sender
                self.assign_edge_type_between(prior, successor)


    # XXX: PAIR: assign_processor()
    def initialize_assigned(self, layer):
        layer.set_pe(None)


    # XXX: PAIR: assign_edge_type_between()
    def initialize_edge_type_between(self, task, successor):
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


    def scheduling_heuristic(self):
        self.initialize_layers_finish_time_zero()
        for app_idx in range(0, len(self.app_list)):
            if app_idx == 0:
                # XXX: original PEFT
                self.peft_algorithm(self.app_list[app_idx], False)
            else:
                # XXX: modified PEFT
                self.peft_algorithm(self.app_list[app_idx], True)
                # self.peft_algorithm(self.app_list[app_idx], False)


    def peft_algorithm(self, target_app, is_modified=False):
        # FIXME: deprecated
        # self._update_vacancy_with_erstwhile_mappings(target_app)

        highest_rank_oct_layers, layers_rank_oct, ready_list = self._init_variables(target_app)

        while ready_list != []:
            # TASK PRIORITIZATION PHASE 
            highest_rank_oct_layer = ready_list[0]
            if highest_rank_oct_layer.is_start_node or highest_rank_oct_layer.is_end_node:
                # XXX: if node is frontend or backend, only cpu processor can be mapped.
                target_pe_list = self.pe_list[:len(config.cpu_config)] 
            else:
                target_pe_list = self.pe_list 

            # PROCESSOR SELECTION PHASE
            min_oeft = float('inf')
            min_oeft_processor = None
            occupation_matrix = self._get_updated_occupation_matrix()
            for processor in target_pe_list:
                oeft = self.compute_oeft(highest_rank_oct_layer, processor)
                if is_modified:
                    # oeft -= self._get_vacancy_by_erstwhile_mappings(highest_rank_oct_layer, processor) # FIXME: deprecated
                    oeft += self.get_interference_from_pe(target_app, processor, occupation_matrix)
                    oeft += self.get_slow_degree_of_pe(highest_rank_oct_layer, processor)
                if oeft < min_oeft: # XXX: processor selection phase
                    min_oeft_processor = processor
                    min_oeft = oeft

            # UPDATING(APPLYING) PHASE
            highest_rank_oct_layer.set_pe(min_oeft_processor)
            highest_rank_oct_layer.set_is_pe_mapped(True)
            self._assign_edge_type_from_priors_to(highest_rank_oct_layer)
            
            self.update_pe_available_time_and_layer_finish_time(highest_rank_oct_layer, min_oeft_processor)

            self.mapped_layers_per_pe[min_oeft_processor.get_idx()].append(highest_rank_oct_layer)
            self.update_ready_list_and_highest_rank_oct_layers(ready_list, layers_rank_oct, highest_rank_oct_layers)


    def _init_variables(self, target_app):
        # init highest_rank_oct_layers, layers_rank_oct
        highest_rank_oct_layers = list()
        layers_rank_oct = dict()
        for layer in target_app.layer_list:
            highest_rank_oct_layers.append(layer)
            layers_rank_oct[layer] = layer.get_rank_oct()

        # init ready_list
        # XXX: insert to ready_list, then extract from highest_rank_oct_layers
        ready_list = list()
        ready_list.append(target_app.layer_list[0])
        highest_rank_oct_layers.remove(target_app.layer_list[0])

        return highest_rank_oct_layers, layers_rank_oct, ready_list


    def compute_oeft(self, layer, processor):
        oct_value = self.optimistic_cost_table[layer.index][processor.idx]
        eft = self.compute_eft(layer, processor)
        return oct_value + eft


    def compute_eft(self, layer, processor):
        est = self.compute_est(layer, processor)
        return est + layer.time_list[processor.idx]


    def compute_est(self, layer, processor):
        in_edges = list(layer.app.graph._in_edge[layer])

        pe_available_time = self.processor_available_time_list[processor.idx]
        max_sum_aft_comm = -float('inf')

        if in_edges != []:
            precedents = list()
            for e in in_edges:
                precedents.append(e.sender)

            for prior in precedents:
                # assign temporarily
                self.assign_processor(layer, processor)
                self.assign_edge_type_between(prior, layer)

                aft = prior.finish_time
                comm_time = prior.app.graph._edge[(prior, layer)].calc_transition_time()
                sum_aft_comm = aft + comm_time
                if sum_aft_comm > max_sum_aft_comm:
                    max_sum_aft_comm = sum_aft_comm

                # initialize temporal assignment
                self.initialize_assigned(layer)
                self.initialize_edge_type_between(prior, layer)

        return max(pe_available_time, max_sum_aft_comm)


    # FIXME: deprecated. Term 'vacancy' may not be used.
    def _update_vacancy_with_erstwhile_mappings(self, target_app):
        self.current_vacancy_per_pe = [target_app.period] * self.num_pe # FIXME: period can be different by app
        for pe_idx, layer_list in enumerate(self.mapped_layers_per_pe):
            for layer in layer_list:
                self.current_vacancy_per_pe[pe_idx] = \
                        self.current_vacancy_per_pe[pe_idx] - layer.time_list[pe_idx]


    # FIXME: deprecated. Term 'vacancy' may not be used.
    def _get_vacancy_by_erstwhile_mappings(self, highest_rank_oct_layer, target_pe):
        penalty = self.current_vacancy_per_pe[target_pe.idx] * config.hyper_parameter
        self.current_vacancy_per_pe[target_pe.idx] = \
                self.current_vacancy_per_pe[target_pe.idx] - highest_rank_oct_layer.time_list[target_pe.idx]

        return penalty


    def get_interference_from_pe(self, target_app, target_pe, occupation_matrix):
        higher_interference = float(0)
        lower_interference = float(0)
        higher_interference_hyper_param = 1
        lower_interference_hyper_param = 1

        for app_idx, app_exec_time in enumerate(occupation_matrix[target_pe.get_idx()]):
            app_priority = app_idx + 1
            if target_app.priority > app_priority: # if there is higher priority app
                higher_interference += app_exec_time
            elif target_app.priority < app_priority: # if there is lower priority app 
                lower_interference += app_exec_time
            elif target_app.priority == app_priority:
                continue

        # result = higher_interference * higher_interference_hyper_param + \
        #         lower_interference * lower_interference_hyper_param
        result = higher_interference * higher_interference_hyper_param

        return result


    def get_slow_degree_of_pe(self, highest_rank_oct_layer, target_pe):
        slow_degree_hyper_param = 1
        result = highest_rank_oct_layer.time_list[target_pe.idx] * slow_degree_hyper_param
        return result


    def _get_updated_occupation_matrix(self):
        # XXX: scheme of occupation matrix below
        # row: PE
        # column: App
        # element: sum of execution time of app on PE

        # init occupation matrix
        occupation_matrix = list()
        for pe_idx in range(self.num_pe):
            occupation_matrix.append(list())
            for _ in range(self.num_app):
                occupation_matrix[pe_idx].append(float(0))

        # update occupation matrix from erstwhile mapping
        for pe_idx in range(self.num_pe):
            layers = self.mapped_layers_per_pe[pe_idx]
            for layer in layers:
                for app_idx, app in enumerate(self.app_list):
                    if layer.app == app:
                        occupation_matrix[pe_idx][app_idx] += layer.time_list[pe_idx]
                        break

        return occupation_matrix


    def update_ready_list_and_highest_rank_oct_layers(self, ready_list, layers_rank_oct, highest_rank_oct_layers):
        del ready_list[0]

        runnable_layers = self.extract_runnable_layers_from(highest_rank_oct_layers)
        for l in runnable_layers: # insert runnable layers to ready_list
            ready_list.append(l)

        ready_list = sorted(ready_list, key=layers_rank_oct.__getitem__, reverse=True) # sort

        for layer in runnable_layers:
            highest_rank_oct_layers.remove(layer)


    def extract_runnable_layers_from(self, highest_rank_oct_layers):
        runnable_layers = list()
        for layer in highest_rank_oct_layers:
            is_runnable = True
            if list(layer.app.graph._in_edge[layer]) != []:
                for in_edges in list(layer.app.graph._in_edge[layer]):
                    if not in_edges.sender.get_is_pe_mapped():
                        is_runnable = False
                        break
            if is_runnable:
                runnable_layers.append(layer)

        return runnable_layers


    def update_pe_available_time_and_layer_finish_time(self, highest_rank_oct_layer, min_oeft_processor):
        in_edges = list(highest_rank_oct_layer.app.graph._in_edge[highest_rank_oct_layer])

        if in_edges == []:
            self.processor_available_time_list[min_oeft_processor.idx] += \
                    round(highest_rank_oct_layer.time_list[min_oeft_processor.idx], 2)
            highest_rank_oct_layer.finish_time = self.processor_available_time_list[min_oeft_processor.idx]
        else:
            max_prior_finish_and_comm_time = -float('inf')
            for e in in_edges:
                self._update_pe_available_time_of_prior(e)
                prior_finish_and_comm_time = e.sender.finish_time
                if prior_finish_and_comm_time > max_prior_finish_and_comm_time:
                    max_prior_finish_and_comm_time = prior_finish_and_comm_time

            self.processor_available_time_list[min_oeft_processor.idx] = \
                    max_prior_finish_and_comm_time + \
                    round(highest_rank_oct_layer.time_list[min_oeft_processor.idx], 2)
            highest_rank_oct_layer.finish_time = self.processor_available_time_list[min_oeft_processor.idx]


    def _update_pe_available_time_of_prior(self, prior_edge):
        prior = prior_edge.sender
        self.processor_available_time_list[prior.pe.get_idx()] += prior_edge.calc_transition_time()
        prior.finish_time = self.processor_available_time_list[prior.pe.get_idx()]


    def initialize_layers_finish_time_zero(self):
        for layer in self.layer_list:
            layer.finish_time = 0


    def get_mappings(self):
        # XXX: mappings = [[mapping], [mapping], ...] # pareto result possible
        mappings = [0] * len(self.layer_list)
        idx = 0
        for app in self.app_list:
            for layer in app.layer_list:
                mappings[idx] = layer.pe.get_idx()
                idx = idx + 1

        print "====================================================="
        for app in self.app_list:
            print app.name, app.layer_list[len(app.layer_list)-1].name, "| finish time:", app.layer_list[len(app.layer_list)-1].finish_time
        print "mapping: ", [mappings]
        print "====================================================="

        return [mappings]


    def print_mapped_layers_on_each_pe(self):
        for i in range(self.num_pe):
            print self.mapped_layers_per_pe[i]


