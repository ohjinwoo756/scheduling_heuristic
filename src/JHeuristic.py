from mapping_function import MapFunc
from fitness import Fitness # calculate mapping's fitness value
from cpu_utilization import CPU_utilization  # TODO: may be applied.
from pe import PEType
import config

class JHeuristic(MapFunc):

    def __init__(self, app_list, pe_list):
        self.app_list = app_list
        self.layer_list = [l for app in app_list for l in app.layer_list]
        self.pe_list = pe_list
        self.num_app = len(app_list)
        self.num_layer = len(self.layer_list)
        self.num_pe = len(pe_list)
        self.fitness = Fitness(app_list, pe_list) # calculate mapping's fitness value

        self.optimistic_cost_table = list()
        self.optimistic_cost_hash = dict() # hash table for speeding up oct calculation


    def do_schedule(self):
        self.make_optimistic_cost_table()
        self.print_optimistic_cost_table() # debug

        self.prioritize_tasks() # task prioritization phase
        self.assign_task_to_processor() # processor selection phase

        # mappings = [net_1_mappings, net_2_mappings, ...]
        # mappings = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2, 0, 0, 0, 0] # mnv1
        # return [mappings] # XXX: [ [], [], [], ... ]: pareto results
        return [] # no solution


    def make_optimistic_cost_table(self):
        for app in self.app_list:
            for layer in app.layer_list:
                oct_layer_row = list() # represents a row(task) in oct (optimistic cost table)
                for pe in self.pe_list:
                    oct_layer_row.append(self.calculate_optimistic_cost(app, layer, pe))
                self.optimistic_cost_table.append(oct_layer_row)


    def calculate_optimistic_cost(self, app, task, processor):
        if (task.name, processor.name) in self.optimistic_cost_hash:
            return self.optimistic_cost_hash[(task.name, processor.name)]
        else:
            if task.is_end_node:
                self.optimistic_cost_hash[(task.name, processor.name)] = 0
                return self.optimistic_cost_hash[(task.name, processor.name)]
            else:
                max_value = -float('inf')
                for successor in task.app.graph._graph[task]:
                    min_value = float('inf')
                    for pe in self.pe_list:
                        self.assign_temporal_processor(task, processor, successor, pe)
                        self.assign_temporal_edge_type_between(task, successor)

                        # debug
                        # if task.pe == successor.pe:
                        #     print task.pe.type, successor.pe.type
                        #     print app.graph._edge[(task, successor)].calc_transition_time()
                        value = self.calculate_optimistic_cost(app, successor, pe) + successor.time_list[pe.idx] + app.graph._edge[(task, successor)].calc_transition_time()

                        self.initialize_temporal_assigned(task, successor)
                        if value < min_value:
                            min_value = value

                    if min_value > max_value:
                        max_value = min_value

                optimistic_cost = max_value
                self.optimistic_cost_hash[(task.name, processor.name)] = optimistic_cost
                return optimistic_cost


    def assign_temporal_processor(self, src, src_pe, dst, dst_pe):
        src.set_pe(src_pe)
        dst.set_pe(dst_pe)


    def assign_temporal_edge_type_between(self, task, successor):
        if task.pe == successor.pe:
            # print task.pe.name
            # print successor.pe.name
            # print task.app.graph._edge[(task, successor)].transition_flag
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


    def initialize_temporal_assigned(self, task, successor):
        task.set_pe(None)
        successor.set_pe(None)
        task.app.graph._edge[(task, successor)].transition_flag = False 
        task.app.graph._edge[(task, successor)].cpu2gpu = False 
        task.app.graph._edge[(task, successor)].gpu2cpu = False 
        task.app.graph._edge[(task, successor)].cpu2npu = False 
        task.app.graph._edge[(task, successor)].npu2cpu = False 
        task.app.graph._edge[(task, successor)].gpu_npu_connection = False


    def print_optimistic_cost_table(self):
        idx = 0
        for app in self.app_list:
            for layer in app.layer_list:
                print self.optimistic_cost_table[idx]
                idx = idx + 1


    def prioritize_tasks(self):
        pass


    def assign_task_to_processor(self):
        pass
