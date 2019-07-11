from heuristic import Heuristic
from sched_simulator import SchedSimulator
from cpu_utilization import CPU_utilization
from deadline import Deadline
from pe import PEType
from pe import PE
import config
from random import SystemRandom


class LayerBasedHeuristic(Heuristic):
    def __init__(self, app_list, pe_list):
        super(LayerBasedHeuristic, self).__init__(app_list, pe_list)
        self.random = SystemRandom()
        self.npu_cutline_percentage = 40  # XXX: Below the percentage, NPU will be assigned
        self.gpu_cutline_percentage = 20  # XXX: Below the percentage, GPU will be assigned

    def _pass_constraint(self, mapping):
        iteration = 0
        trial_num = 100
        while iteration < trial_num:
            if self._apply_deadline_cst(mapping):
                return True
            iteration += 1
        return False

    def _set_initial_mapping(self, mapping):
        criteria_list = self._init_criteria_list()
        self._do_map(criteria_list, mapping)

    def _init_criteria_list(self):
        criteria_list = []
        for idx, layer in enumerate(self.layer_list):
            self._do_init_list(criteria_list, idx, layer)
        return criteria_list

    def _do_init_list(self, criteria_list, idx, layer):
        obj_value_on_cpu = layer.time_list[PEType.CPU]
        obj_value_on_gpu = layer.time_list[PEType.CPU + config.num_virtual_cpu]
        if config.opt_energy:
            obj_value_on_cpu += layer.time_list[PEType.CPU]
            obj_value_on_gpu += layer.time_list[PEType.CPU + config.num_virtual_cpu]
        gap_percentage = (obj_value_on_gpu / obj_value_on_cpu) * 100
        criteria_list.append(gap_percentage)

    def _do_map(self, criteria_list, mapping):
        for idx in range(0, self.num_layer):
            if criteria_list[idx] < self.gpu_cutline_percentage:
                mapping[idx] = PEType.CPU + config.num_virtual_cpu
            elif criteria_list[idx] < self.npu_cutline_percentage:
                mapping[idx] = PEType.CPU + config.num_virtual_cpu + 1
            else:
                mapping[idx] = self.random.randint(0, config.num_virtual_cpu - 1)

    def _apply_deadline_cst(self, mapping):
        csts = []
        PE.init_apps_pe_by_mapping(self.app_list, mapping, self.pe_list)
        for idx, app in enumerate(self.app_list):
            if config.app_to_cst_dict[idx] == 'Deadline':
                csts.append(Deadline(self.app_list, app, self.pe_list))

        # if there is deadline constraint
        for idx, cst in enumerate(csts):
            penalty = cst.constraint_function(mapping)[0]
            if penalty != 0:
                self._one_bit_flip_local_optimization(mapping, penalty, cst)
                return False

        return True

    def _one_bit_flip_local_optimization(self, mapping, penalty, cst):
        sigma = range(0, self.num_layer)
        improved = True
        while improved:
            improved = False
            for sig_idx in sigma:
                if self._calc_delta(sig_idx, mapping, penalty, cst):
                    improved = True

    def _calc_delta(self, sig_idx, ind, penalty, cst):
        prev_cost = penalty
        prev_processor_index = ind[sig_idx]

        num_virtual_cpu = config.num_virtual_cpu

        if (sig_idx + 1) in config.start_nodes_idx and ind[sig_idx + 1] >= num_virtual_cpu:
            return False

        if (sig_idx + 1) in config.end_nodes_idx and ind[sig_idx - 1] >= num_virtual_cpu:
            return False

        if sig_idx > 0 and sig_idx < self.num_layer - 1 and ind[sig_idx] != ind[sig_idx - 1] and ind[sig_idx] != ind[sig_idx + 1]:
            ind[sig_idx] = ind[sig_idx - 1]
            next_value_1 = cst.constraint_function(ind)[0]

            ind[sig_idx] = ind[sig_idx + 1]
            next_value_2 = cst.constraint_function(ind)[0]
            if next_value_1 < next_value_2:
                next_value = next_value_1
                ind[sig_idx] = ind[sig_idx - 1]
            else:
                next_value = next_value_2
                ind[sig_idx] = ind[sig_idx + 1]
        elif sig_idx > 0 and ind[sig_idx] != ind[sig_idx - 1]:
            ind[sig_idx] = ind[sig_idx - 1]
            next_value = cst.constraint_function(ind)[0]
        elif sig_idx < self.num_layer - 1 and ind[sig_idx] != ind[sig_idx + 1]:
            ind[sig_idx] = ind[sig_idx + 1]
            next_value = cst.constraint_function(ind)[0]
        else:
            return False

        next_cost = next_value

        result = prev_cost - next_cost
        if result > 0:
            penalty = next_value
            return True
        else:
            ind[sig_idx] = prev_processor_index
            return False
