

from mapping_function import MapFunc
from fitness import Fitness
from pe import PEType
from collections import defaultdict
from Layer import Layer
from sched_simulator import SchedSimulator
import config
import copy


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
        self.rank_of_img_pe = None
        self.img_pe = list()
        self.num_of_img_pe = 0

        # row = PE, col = App
        # content = [-1/1, absolute difference]
        self.perf_improv_per_app = [[0] * self.num_app for _ in range(self.num_pe)]
        self.sum_of_perf_per_pe = [0] * self.num_pe
        self.new_moving_layers = None
        self.new_degrading_layers = [None] * self.num_pe
        self.degrading_pe = [-1] * self.num_pe

        self.solutions = []


    def do_schedule(self):
        self.heuristic()
        return self.solutions 


    def heuristic(self):
        # apply PEFT to each application
        self.calculate_oct_and_rank_oct()
        temp_list = [[] for _ in range(self.num_pe)]
        for app_idx, app in enumerate(self.app_list):
            self.initialize_variables_for_peft()
            self.peft_algorithm(app, len(app.layer_list))

            for pe in self.pe_list:
                pe_idx = pe.get_idx()
                for l in self.mapped_layers_per_pe[pe_idx]:
                    temp_list[pe_idx].append(l)
        self.mapped_layers_per_pe = temp_list


        if self.num_app == 1:
            self.solutions.append(self.get_mappings()[0])
        else:
            self.rank_processors()
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
                        # assign PE temporarily (this is for getting transition time below)
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


    # PAIR: initialize_assigned()
    def assign_processor(self, layer, pe):
        layer.set_pe(pe)


    # PAIR: initialize_edge_type_between()
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


    # PAIR: assign_processor()
    def initialize_assigned(self, layer):
        layer.set_pe(None)


    # PAIR: assign_edge_type_between()
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


    def peft_algorithm(self, target_app, coverage):
        highest_rank_oct_layers, layers_rank_oct, ready_list = self._init_variables(target_app, coverage)

        while ready_list != []:
            # ---- TASK PRIORITIZATION PHASE ----
            highest_rank_oct_layer = ready_list[0]
            if highest_rank_oct_layer.is_start_node or highest_rank_oct_layer.is_end_node:
                # if node is frontend or backend, only cpu processor can be mapped.
                target_pe_list = self.pe_list[:len(config.cpu_config)] 
            else:
                target_pe_list = self.pe_list 

            # ---- PROCESSOR SELECTION PHASE ----
            min_oeft = float('inf')
            min_oeft_processor = None
            occupation_matrix = self._get_updated_occupation_matrix()
            for processor in target_pe_list:
                oeft = self.compute_oeft(highest_rank_oct_layer, processor)

                if oeft < min_oeft:
                    min_oeft_processor = processor
                    min_oeft = oeft

            # ---- UPDATING(APPLYING) PHASE ----
            highest_rank_oct_layer.set_pe(min_oeft_processor)
            highest_rank_oct_layer.set_is_pe_mapped(True)
            self._assign_edge_type_from_priors_to(highest_rank_oct_layer)
            
            self.update_pe_available_time_and_layer_finish_time(highest_rank_oct_layer, min_oeft_processor)

            self.mapped_layers_per_pe[min_oeft_processor.get_idx()].append(highest_rank_oct_layer)
            self.update_ready_list_and_highest_rank_oct_layers(ready_list, layers_rank_oct, highest_rank_oct_layers)


    def _init_variables(self, target_app, coverage):
        highest_rank_oct_layers = list()
        layers_rank_oct = dict()

        start_point = self.progress_by_app[target_app.priority-1]
        coveraged_layer_list = target_app.layer_list[start_point:start_point+coverage]

        for layer in coveraged_layer_list:
            highest_rank_oct_layers.append(layer)
            layers_rank_oct[layer] = layer.get_rank_oct()

        # initialize 
        ready_list = list()
        ready_list.append(coveraged_layer_list[0])
        highest_rank_oct_layers.remove(coveraged_layer_list[0])

        self.progress_by_app[target_app.priority-1] += coverage

        return highest_rank_oct_layers, layers_rank_oct, ready_list


    def _get_updated_occupation_matrix(self):
        # ---- scheme of occupation matrix below ----
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


    def _assign_edge_type_from_priors_to(self, successor):
        in_edges = list(successor.app.graph._in_edge[successor])
        if in_edges != []:
            for e in in_edges:
                prior = e.sender
                self.assign_edge_type_between(prior, successor)


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


    def update_ready_list_and_highest_rank_oct_layers(self, ready_list, layers_rank_oct, highest_rank_oct_layers):
        del ready_list[0]

        runnable_layers = self.extract_runnable_layers_from(highest_rank_oct_layers)
        for l in runnable_layers:
            ready_list.append(l)

        ready_list = sorted(ready_list, key=layers_rank_oct.__getitem__, reverse=True)

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


    # get all applications' layer mapping
    def get_mappings(self):
        mappings = [0] * len(self.layer_list)
        idx = 0
        for app in self.app_list:
            for layer in app.layer_list:
                mappings[idx] = layer.pe.get_idx()
                idx = idx + 1

        return [mappings]


    # TODO too many comments.. fix function / variable name
    def rank_processors(self):
        sample_app = self.app_list[0]
        dict_pe_to_sum = dict()
        dict_img_pe_to_sum = dict()

        # get summation of layer's execution time for each PE
        for p_idx, p in enumerate(self.pe_list):
            exec_sum = 0

            # analyze all layer on the PE, not specific one layer
            for l in sample_app.layer_list:
                # XXX Exception because non-img-processor has very large execution time for frontend and backend layer
                if not l.is_start_node and not l.is_end_node:
                    exec_sum += l.time_list[p_idx]

            # processors will be ranked by execution sum
            dict_pe_to_sum[p_idx] = exec_sum

            # rank image processors
            if p.get_type() == PEType.CPU:
                dict_img_pe_to_sum[p_idx] = exec_sum
                self.img_pe.append(p)
                self.num_of_img_pe += 1

        # ------ sorted upward by sum ------
        # list index    = rank
        # content       = processor's index
	self.rank_of_pe = sorted(dict_pe_to_sum, key=lambda k : dict_pe_to_sum[k])
	self.rank_of_img_pe = sorted(dict_img_pe_to_sum, key=lambda k : dict_img_pe_to_sum[k])


    def peft_synthesis(self):
        initial_mapping = copy.deepcopy(self.get_mappings()[0])
        self.get_solution_if_schedulable(initial_mapping)
        prev_result_tuple = self.fitness.calculate_fitness(initial_mapping)

        # move layers in apps from highest to lowest priority
        for app in self.app_list: 
            progress = 1 # frontend layer excluded
            while True:
                self.perf_improv_per_app = [[0] * self.num_app for _ in range(self.num_pe)]
                self.sum_of_perf_per_pe = [0] * self.num_pe

                if progress == len(app.layer_list)-1: # backend layer excluded
                    # self.print_chunk_progress_with_messsage("[FULL PROGRESS] MOVE TO NEXT APPLICATION", progress, None)
                    break

                chunk_unit_list = [1, 5, 10]
                whether_go_to_next_app, progress, prev_result_tuple = \
                        self.test_on_chunk_unit_list(chunk_unit_list, app, progress, prev_result_tuple)

                if whether_go_to_next_app:
                    break


    def print_chunk_progress_with_messsage(self, message, progress, chunk):
        print "*************************"
        print message
        print "\tPROGRESS :", progress
        print "\tCHUNK :", chunk


    def get_solution_if_schedulable(self, mapping):
        if self.is_schedulable(mapping):
            self.solutions.append(mapping)


    def is_schedulable(self, mapping):
        csts, objs = SchedSimulator._get_csts_and_objs(self.fitness, mapping)
        available_results = True
        for idx, (cst, value) in enumerate(zip(self.fitness.csts, csts)):
            if value[0] != 0:
                available_results = False
                break
        config.available_results = available_results
        return available_results


    # XXX IMPORTANT variables:
    #   - self.perf_improv_per_app
    #   - self.sum_of_perf_per_pe
    def test_on_chunk_unit_list(self, chunk_unit_list, app, progress, prev_result_tuple):
        whether_go_to_next_app = False

        for chunk in chunk_unit_list:
            temp_progress, temp_moving_layers = self.update_variables(app, chunk, progress)

            for target_pe in self.rank_of_pe[:]:
                self.calc_perf_improv_on_target_pe(app, temp_moving_layers, target_pe, prev_result_tuple)
            max_perf_improv_pe = self.select_the_best_perf_improv_among_pe()

            if not self.is_passable(max_perf_improv_pe):
                if chunk == chunk_unit_list[-1]:
                    # self.print_chunk_progress_with_messsage("[UNPASSABLE] move to NEXT APPLICATION", temp_progress, chunk)
                    whether_go_to_next_app = True
                    return whether_go_to_next_app, None, prev_result_tuple
                else:
                    # self.print_chunk_progress_with_messsage("[UNPASSABLE] move to NEXT CHUNK UNIT", temp_progress, chunk)
                    self.perf_improv_per_app = [[0] * self.num_app for _ in range(self.num_pe)]
                    self.sum_of_perf_per_pe = [0] * self.num_pe
                    continue
            else:
                progress = temp_progress # finalize progress
                self.move_to(self.new_moving_layers, max_perf_improv_pe)
                self.move_to(self.new_degrading_layers[max_perf_improv_pe.get_idx()], \
                        self.pe_list[self.degrading_pe[max_perf_improv_pe.get_idx()]])
                new_mapping = copy.deepcopy(self.get_mappings()[0])
                self.get_solution_if_schedulable(new_mapping)
                prev_result_tuple = self.fitness.calculate_fitness(new_mapping) # update previous result

                # self.print_chunk_progress_with_messsage("[PASSABLE] move to NEXT PROGRESS", progress, chunk)
                return whether_go_to_next_app, progress, prev_result_tuple


    # frontend, backend excluded here
    def update_variables(self, app, chunk, progress):
        if progress + chunk > len(app.layer_list)-1:
            new_chunk = len(app.layer_list)-1 - progress
            moving_layers = app.layer_list[progress:progress + new_chunk]
            progress += new_chunk
        else:
            moving_layers = app.layer_list[progress:progress + chunk]
            progress += chunk

        return progress, moving_layers


    def calc_perf_improv_on_target_pe(self, app, moving_layers, target_pe, prev_result_tuple):
        initial_mappings = copy.deepcopy(self.get_mappings()[0])
        initial_mapped_layers_per_pe = self.deepcopy_list(self.mapped_layers_per_pe)

        self.new_moving_layers = moving_layers
        self.move_to(moving_layers, self.pe_list[target_pe])
        # self.move_fb_if_img_pe(app, target_pe)

        new_mapping = None
        other_app_layers = self.get_other_app_in_target(app, target_pe)

        if other_app_layers != []:
            self.new_degrading_layers[target_pe] = other_app_layers[:]
            # XXX return results of tuple for 3 cases
            result_tuples_by_cases, result_mapping, moving_cases = self.degrade_other_layers_to(app, other_app_layers, target_pe)

            # select the best case
            max_perf_improv = -float("inf")
            for case_idx, result in enumerate(result_tuples_by_cases):
                if result == None:
                    continue

                sum_of_perf_improv, perf_improv = self.calculate_WCRT_on_each_app(result, prev_result_tuple)
                if sum_of_perf_improv > max_perf_improv:
                    new_mapping = result_mapping[case_idx][:]
                    self.degrading_pe[target_pe] = moving_cases[case_idx]
        else:
            new_mapping = copy.deepcopy(self.get_mappings()[0])

        result_tuple = self.fitness.calculate_fitness(new_mapping)
        sum_of_perf_improv, perf_improv = self.calculate_WCRT_on_each_app(result_tuple, prev_result_tuple)
        self.perf_improv_per_app[target_pe] = perf_improv
        self.sum_of_perf_per_pe[target_pe] = sum_of_perf_improv

        self.initialize_move(initial_mappings, initial_mapped_layers_per_pe)


    def calculate_WCRT_on_each_app(self, result_tuple, prev_result_tuple):
        sum_of_perf_improv = 0
        perf_improv = list()

        for app_idx, value in enumerate(result_tuple):
            abs_diff = abs(value - prev_result_tuple[app_idx])
            if value < prev_result_tuple[app_idx]:
                perf_improv.append([1, abs_diff]) # performance increases
            else:
                perf_improv.append([-1, abs_diff]) # performance decreases
            sum_of_perf_improv += perf_improv[app_idx][0] * perf_improv[app_idx][1]

        return sum_of_perf_improv, perf_improv


    def deepcopy_list(self, mapped_layers_per_pe):
        initial_mapped_layers_per_pe = list()
        for pe in self.pe_list:
            initial_mapped_layers_per_pe.append(list())
            initial_mapped_layers_per_pe[pe.get_idx()] = list(mapped_layers_per_pe[pe.get_idx()])

        return initial_mapped_layers_per_pe


    def move_to(self, moving_layers, processor):
        if moving_layers != None:
            for l in moving_layers:
                self.mapped_layers_per_pe[l.pe.get_idx()].remove(l)
                self.mapped_layers_per_pe[processor.get_idx()].append(l)
                self.assign_processor(l, processor)


    def move_fb_if_img_pe(self, app, target_pe):
        if self.pe_list[target_pe] in self.img_pe:
            moving_layers = [app.layer_list[0], app.layer_list[-1]]
            self.move_to(moving_layers, self.pe_list[target_pe])
            self.new_moving_layers.append(app.layer_list[0])
            self.new_moving_layers.append(app.layer_list[-1])
        else:
            pass


    def get_other_app_in_target(self, app, target_pe):
        other_app_layers = []
        for layer in self.mapped_layers_per_pe[target_pe]:
            if app.get_priority() < layer.app.get_priority():
                other_app_layers.append(layer)
        return other_app_layers


    def degrade_other_layers_to(self, app, other_app_layers, target_pe):
        result_tuples_by_cases = list()
        result_mapping = list()
        faster_pe_idx, slower_pe_idx, moving_cases = self.search_faster_slower_pe(target_pe)

        # mapping after primary moves and before degrading phase
        inter_mappings = copy.deepcopy(self.get_mappings()[0])
        inter_mapped_layers_per_pe = self.deepcopy_list(self.mapped_layers_per_pe)

        # ------- DEGRADING PHASE -------
        # case 1: no change
        case_one_mapping = inter_mappings
        result_tuples_by_cases.append(self.fitness.calculate_fitness(case_one_mapping))
        # result_tuples_by_cases.append([99999999] * self.num_app)
        result_mapping.append(case_one_mapping)

        # case 2: move to faster PE
        if faster_pe_idx != -1:
            self.move_to(other_app_layers, self.pe_list[faster_pe_idx])
            case_two_mapping = copy.deepcopy(self.get_mappings()[0])
            result_tuples_by_cases.append(self.fitness.calculate_fitness(case_two_mapping))
            # result_tuples_by_cases.append([99999999] * self.num_app)
            result_mapping.append(case_two_mapping)
            self.initialize_move(inter_mappings, inter_mapped_layers_per_pe)
        else:
            result_tuples_by_cases.append(None)
            result_mapping.append(None)

        # case 3: move to slower PE
        if slower_pe_idx != -1:
            self.move_to(other_app_layers, self.pe_list[slower_pe_idx])
            case_three_mapping = copy.deepcopy(self.get_mappings()[0])
            result_tuples_by_cases.append(self.fitness.calculate_fitness(case_three_mapping))
            result_mapping.append(case_three_mapping)
            self.initialize_move(inter_mappings, inter_mapped_layers_per_pe)
        else:
            result_tuples_by_cases.append(None)
            result_mapping.append(None)

        return result_tuples_by_cases, result_mapping, moving_cases


    def search_faster_slower_pe(self, target_pe):
        faster_pe_idx = -1
        slower_pe_idx = -1
        for idx, pe in enumerate(self.rank_of_pe):
            if pe == target_pe:
                if idx-1 >= 0:
                    faster_pe_idx = self.rank_of_pe[idx-1]
                if idx+1 < self.num_pe:
                    slower_pe_idx = self.rank_of_pe[idx+1]
        moving_cases = [target_pe, faster_pe_idx, slower_pe_idx]

        return faster_pe_idx, slower_pe_idx, moving_cases


    def initialize_move(self, initial_mappings, initial_mapped_layers_per_pe):
        l_idx = 0
        for app_idx, app in enumerate(self.app_list):
            for l in app.layer_list:
                self.assign_processor(l, self.pe_list[initial_mappings[l_idx]])
                l_idx += 1
        self.mapped_layers_per_pe = self.deepcopy_list(initial_mapped_layers_per_pe)


    def select_the_best_perf_improv_among_pe(self):
        max_perf_improv = -float("inf")

        for pe in self.rank_of_pe[:]:
            if self.sum_of_perf_per_pe[pe] > max_perf_improv:
                max_perf_improv = self.sum_of_perf_per_pe[pe]
                max_perf_improv_pe = self.pe_list[pe]
        return max_perf_improv_pe


    def is_passable(self, max_perf_improv_pe):
        passable = False
        for app_idx, app in enumerate(self.app_list):
            if self.perf_improv_per_app[max_perf_improv_pe.get_idx()][app_idx][0] == 1:
                # if any app performs better than before, it's passable. 
                passable = True
        return passable


    def print_mapped_layers_on_each_pe(self):
        for i in range(self.num_pe):
            temp = []
            for l in self.mapped_layers_per_pe[i]:
                temp.append(l.index)
            print temp
        print ""


    def print_initial(self, initial_mapped_layers_per_pe):
        for i in range(self.num_pe):
            temp = []
            for l in initial_mapped_layers_per_pe[i]:
                temp.append(l.index)
            print temp
        print ""


    def print_optimistic_cost_table(self):
        table_row_idx = 0
        for app in self.app_list:
            for layer in app.layer_list:
                print self.optimistic_cost_table[table_row_idx]
                table_row_idx = table_row_idx + 1


