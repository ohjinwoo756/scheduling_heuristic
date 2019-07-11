from objective import Objective
from throughput import Throughput
from response_time import Response_time
from energy_consumption import Energy_consumption
from cpu_utilization import CPU_utilization
from deadline import Deadline
from pe import PE
import config


class Fitness():
    def __init__(self, app_list, pe_list):
        self.app_list = app_list
        self.app_num = len(app_list)
        self.pe_list = pe_list
        self.fitness = 0.0
        self.best_fitness = float("inf")
        self.objs = []
        self.csts = []

        self._init_obj_instances_by_app()
        self._init_cst_instances_by_app()

        if config.CPU_UTILIZATION != 100:
            self.add_whole_constraint('CPU_utilization')

        if config.opt_energy:
            self.add_whole_objective('Energy_consumption')

        if config.energy_cst != 0:
            self.add_whole_constraint('Energy_consumption')

    def _init_obj_instances_by_app(self):
        for idx, app in enumerate(self.app_list):
            if config.app_to_obj_dict[idx] == 'Response_time':
                self.objs.append(Response_time(self.app_list, app, self.pe_list))
            elif config.app_to_obj_dict[idx] == 'Throughput':
                self.objs.append(Throughput(self.app_list, app, self.pe_list))
            app.set_obj(self.objs[-1])

    def _init_cst_instances_by_app(self):
        for idx, app in enumerate(self.app_list):
            if config.app_to_cst_dict[idx] == 'Deadline':
                self.csts.append(Deadline(self.app_list, app, self.pe_list))
            elif config.app_to_cst_dict[idx] == 'None':
                self.csts.append(None)
            app.set_cst(self.csts[-1])

    def add_whole_objective(self, objective):
        if objective == 'Energy_consumption':
            self.objs.append(Energy_consumption(self.app_list, None, self.pe_list))

    def add_whole_constraint(self, constraint):
        if constraint == 'CPU_utilization':
            self.csts.append(CPU_utilization(self.app_list, None, self.pe_list))
        elif constraint == 'Energy_consumption':
            self.csts.append(Energy_consumption(self.app_list, None, self.pe_list))

    # This method is called when GA module calculates fitness value
    # Mapping information is only passed by this method
    def calculate_fitness(self, mapping):
        PE.init_apps_pe_by_mapping(self.app_list, mapping, self.pe_list)

        objs = []
        for obj in self.objs:
            obj_temp = obj.objective_function(mapping)
            objs.append(obj_temp[0])

        for idx, cst in enumerate(self.csts):
            if cst is not None and isinstance(cst, Deadline):
                cst_temp = cst.constraint_function(mapping)
                objs[idx] += cst_temp[0]
            elif cst is not None:
                cst_temp = cst.constraint_function(mapping)
                objs = [v + cst_temp[0] for v in objs]

        return tuple(objs)
