
import config
from objective import Objective
from constraint import Constraint
from pe import PEType


class Energy_consumption(Objective, Constraint):
    def __init__(self, app_list, app, pe_array):
        super(Energy_consumption, self).__init__(app_list, app, pe_array)
        self.penalty_alpha = 200
        self.total_cpu_core_num = self.get_total_cpu_core_num() 

    def get_total_cpu_core_num(self):
        total_cpu_core_num = 0
        for cpu_pe in range(0, len(config.cpu_config)):
            core_num = int(config.cpu_config[cpu_pe])
            total_cpu_core_num += core_num
        return total_cpu_core_num

    def _get_energy(self):
        energy = 0.0
        for app in self.app_list:
            one_app_energy = 0.0
            for layer in app.layer_list:
                pe = layer.get_pe()
                exec_time = layer.time_list[pe.get_idx()]
                if pe.get_type() == PEType.CPU:
                    hyper_param_by_cpu_core = pe.get_cpu_core_num() / float(self.total_cpu_core_num)
                    one_app_energy += exec_time * pe.get_energy() * hyper_param_by_cpu_core
                else:
                    one_app_energy += exec_time * pe.get_energy()
            energy += one_app_energy
        return energy

    def _objective_function(self, mapping):
        del mapping # unused 
        return self._get_energy(),

    def _constraint_function(self):
        energy = self._get_energy()
        penalty_decision = energy - float(config.energy_cst)
        penalty = self.penalty_alpha * max(penalty_decision, 0)
        return penalty, energy
