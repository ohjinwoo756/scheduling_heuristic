
from mapping_function import MapFunc
from fitness import Fitness
from pe import PEType
from collections import defaultdict
from Layer import Layer
from sched_simulator import SchedSimulator
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
        self.progress_by_app = [0] * self.num_app

        self.rank_of_pe = None
        self.rank_of_cpu_pe = None
        self.num_of_cpu_pe = 0
        self.solutions = []


    def do_schedule(self):
        self.synthetic_heuristic()
        # self.print_mapped_layers_on_each_pe()
        return self.solutions 


    def synthetic_heuristic(self):
        # XXX: apply PEFT to each application
        self.calculate_oct_and_rank_oct()
        for app_idx, app in enumerate(self.app_list):
            self.initialize_variables_for_peft()
            self.peft_algorithm(app, len(app.layer_list), False)

        # XXX: In case of multiple PEFTs, 
        if self.num_app == 1:
            self.solutions.append(self.get_mappings()[0])
        else:
            self.rank_processors()
            # self.initial_synthesis()
            self.peft_synthesis()


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


    def initialize_variables_for_peft(self):
        for layer in self.layer_list:
            layer.finish_time = 0
        self.processor_available_time_list = [0] * self.num_pe
        self.mapped_layers_per_pe = [[] for _ in range(self.num_pe)]


    # XXX: target_app.layer_list[self.progress_by_app[target_app.priority-1] : coverage] will be processed
    def peft_algorithm(self, target_app, coverage, is_modified=False):
        highest_rank_oct_layers, layers_rank_oct, ready_list = self._init_variables(target_app, coverage)

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
                    oeft += self.get_interference_from_pe(target_app, processor, occupation_matrix)

                if oeft < min_oeft:
                    min_oeft_processor = processor
                    min_oeft = oeft

            # UPDATING(APPLYING) PHASE
            highest_rank_oct_layer.set_pe(min_oeft_processor)
            highest_rank_oct_layer.set_is_pe_mapped(True)
            self._assign_edge_type_from_priors_to(highest_rank_oct_layer)
            
            self.update_pe_available_time_and_layer_finish_time(highest_rank_oct_layer, min_oeft_processor)

            self.mapped_layers_per_pe[min_oeft_processor.get_idx()].append(highest_rank_oct_layer)
            self.update_ready_list_and_highest_rank_oct_layers(ready_list, layers_rank_oct, highest_rank_oct_layers)


    def _init_variables(self, target_app, coverage):
        # init highest_rank_oct_layers, layers_rank_oct
        highest_rank_oct_layers = list()
        layers_rank_oct = dict()

        start_point = self.progress_by_app[target_app.priority-1]
        coveraged_layer_list = target_app.layer_list[start_point:start_point+coverage]

        for layer in coveraged_layer_list:
            highest_rank_oct_layers.append(layer)
            layers_rank_oct[layer] = layer.get_rank_oct()

        # init ready_list
        # XXX: insert to ready_list, then extract from highest_rank_oct_layers
        ready_list = list()
        ready_list.append(coveraged_layer_list[0])
        highest_rank_oct_layers.remove(coveraged_layer_list[0])

        self.progress_by_app[target_app.priority-1] += coverage

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


    def rank_processors(self):
        sample_app = self.app_list[0]
        dict_pe_to_sum = dict()
        dict_cpu_pe_to_sum = dict()

        # get sum of layer's execution time for each PE
        for p_idx, p in enumerate(self.pe_list):
            exec_sum = 0
            for l in sample_app.layer_list:
                if not l.is_start_node and not l.is_end_node:
                    exec_sum += l.time_list[p_idx]
            dict_pe_to_sum[p_idx] = exec_sum
            if p.get_type() == PEType.CPU:
                dict_cpu_pe_to_sum[p_idx] = exec_sum
                self.num_of_cpu_pe += 1

        # sorted upward by sum
        # XXX: index = rank, content = processor's index
	self.rank_of_pe = sorted(dict_pe_to_sum, key=lambda k : dict_pe_to_sum[k])
	self.rank_of_cpu_pe = sorted(dict_cpu_pe_to_sum, key=lambda k : dict_cpu_pe_to_sum[k])


    # FIXME: deprecated
    # def initial_synthesis(self):
    #     for a_idx, a in enumerate(self.app_list):
    #         max_parallel = self.get_max_parallelism_of(a)
    #         if max_parallel >= self.num_pe:
    #             shrink_parallelism(a)


    # FIXME: deprecated
    # def get_max_parallelism_of(self, app):
    #     max_parallel_of_app = 1
    #     for l_idx , l in enumerate(app.layer_list):
    #         max_parallel = 0
    #         out_edges = l.app.graph._out_edge[l]
    #         if len(out_edges) >= 2: # if parallel structure
    #             mapped = []
    #             for e in out_edges:
    #                 rp = e.receiver.get_pe()
    #                 if rp not in mapped:
    #                     mapped.append(rp)
    #                     max_parallel += 1
    #         if max_parallel > max_parallel_of_app:
    #             max_parallel_of_app = max_parallel

    #     return max_parallel_of_app


    # FIXME: deprecated
    # def shrink_parallelism(self, app):
    #     for l_idx, l in enumerate(app.layer_list):
    #         if l.get_pe() == self.rank_of_pe[-1]:
    #             l.set_pe(self.rank_of_pe[0])


    def peft_synthesis(self):
        # XXX: get WCRT from initial mapping
        initial_mapping = self.get_mappings()[0]
        init_res_tuple = self.fitness.calculate_fitness(initial_mapping)
        if self.is_schedulable(initial_mapping):
            self.solutions.append(initial_mapping)

        prev_res_tuple = init_res_tuple
        chunk = 5

        # XXX: from the highest to lowest priority application
        for app in self.app_list: 
            progress = 0

            while progress <= len(app.layer_list):
                moving_layers = app.layer_list[progress:progress+chunk]
                perf_improv_per_app = [[0] * self.num_app for _ in range(self.num_pe)]
                sum_of_perf_per_pe = [0] * self.num_pe
                max_perf_improv = -float("inf")
                max_perf_improv_pe = None
                passable = False

                # from the 2nd fast processor
                for pe in self.rank_of_pe[1:]:
                    self.move_to(moving_layers, self.pe_list[pe]) # temporarily
                    mapping = self.get_mappings()[0]
                    res_tuple = self.fitness.calculate_fitness(mapping)
                    for app_idx, res in enumerate(res_tuple):
                        abs_diff = abs(res - prev_res_tuple[app_idx])
                        if res < prev_res_tuple[app_idx]:
                            perf_improv_per_app[pe][app_idx] = [1, abs_diff]
                        else:
                            perf_improv_per_app[pe][app_idx] = [-1, abs_diff]
                        sum_of_perf_per_pe[pe] += perf_improv_per_app[pe][app_idx][0] * \
                                                    perf_improv_per_app[pe][app_idx][1]

                    if sum_of_perf_per_pe[pe] > max_perf_improv:
                        max_perf_improv = sum_of_perf_per_pe[pe]
                        max_perf_improv_pe = self.pe_list[pe]
                        for app_idx, app in enumerate(self.app_list):
                            if perf_improv_per_app[pe][app_idx][0] == 1:
                                passable = True

                    self.initialize_move(moving_layers, self.pe_list[pe]) # initialize

                self.move_to(moving_layers, max_perf_improv_pe) # final assignment
                mapping = self.get_mappings()[0]
                res_tuple = self.fitness.calculate_fitness(mapping)
                prev_res_tuple = res_tuple

                if not passable:
                    break
                if self.is_schedulable(mapping):
                    self.solutions.append(mapping)

                progress += chunk 


    def is_schedulable(self, mapping):
        csts, objs = SchedSimulator._get_csts_and_objs(self.fitness, mapping)
        available_results = True
        for idx, (cst, value) in enumerate(zip(self.fitness.csts, csts)):
            if value[0] != 0:
                available_results = False
                break
        config.available_results = available_results

        return available_results


    def move_to(self, moving_layers, processor):
        for l in moving_layers:
            self.assign_processor(l, processor)


    def initialize_move(self, moving_layers, processor):
        pass


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

        result = higher_interference * higher_interference_hyper_param + \
                lower_interference * lower_interference_hyper_param

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


    def get_mappings(self):
        # XXX: mappings = [[mapping], [mapping], ...] # pareto result possible
        mappings = [0] * len(self.layer_list)
        idx = 0
        for app in self.app_list:
            for layer in app.layer_list:
                mappings[idx] = layer.pe.get_idx()
                idx = idx + 1

        # print "====================================================="
        # for app in self.app_list:
        #     print app.name, app.layer_list[len(app.layer_list)-1].name, "| finish time:", app.layer_list[len(app.layer_list)-1].finish_time
        # print "mapping: ", [mappings]
        # print "====================================================="

        return [mappings]


    def print_mapped_layers_on_each_pe(self):
        for i in range(self.num_pe):
            print self.mapped_layers_per_pe[i]


    def print_optimistic_cost_table(self):
        table_row_idx = 0
        for app in self.app_list:
            for layer in app.layer_list:
                print self.optimistic_cost_table[table_row_idx]
                table_row_idx = table_row_idx + 1

