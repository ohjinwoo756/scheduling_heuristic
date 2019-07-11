from random import SystemRandom
from heuristic import Heuristic
from sched_simulator import SchedSimulator
from cpu_utilization import CPU_utilization
from pe import PEType
from pe import PE
import config


class AppBasedHeuristic(Heuristic):
    def __init__(self, app_list, pe_list):
        super(AppBasedHeuristic, self).__init__(app_list, pe_list)
        self.random = SystemRandom()
        self.cur_map_stat_per_pe = None

    def _pass_constraint(self):
        iteration = 0
        trial_num = 1000
        prev_mapping = list(self.mapping)
        while iteration < trial_num:
            # self._apply_cpu_util_cst(mapping)
            print("[{0:^5d}]\nMapping: {1}\n".format(iteration, self.mapping))
            if self.find_schedulable_mapping():
                return True
            if self.mapping == prev_mapping:
                break
            prev_mapping = list(self.mapping)
            iteration += 1
        return False

    def _set_initial_mapping(self):
        obj_val_matrix = self._init_obj_val_matrix()
        self._do_map(obj_val_matrix)

    def _init_obj_val_matrix(self):
        matrix = dict()
        for app in self.app_list:
            matrix[app.priority] = dict()
            for pe in self.pe_list:
                self._do_init_matrix(matrix, app, pe)
        return matrix

    def _do_init_matrix(self, matrix, app, pe):
        obj_val = 0
        for layer in app.layer_list:
            if not layer.is_start_node and not layer.is_end_node:
                obj_val += layer.time_list[pe.get_idx()]
                if config.opt_energy:
                    obj_val += layer.time_list[pe.get_idx()] * pe.get_energy()
        matrix[app.priority][pe.name] = obj_val

    def _do_map(self, obj_val_matrix):
        prev_mapped_pe = None
        self.cur_map_stat_per_pe = self._init_current_map_status()

        for prio in range(1, len(self.app_list) + 1):
            best_value, best_pe = self._get_the_best(obj_val_matrix[prio])
            if prev_mapped_pe == best_pe.name:
                reduced_dict = obj_val_matrix[prio]
                reduced_dict[best_pe.name] = float("inf")
                second_value, second_pe = self._get_the_best(reduced_dict)
                if self.cur_map_stat_per_pe[prev_mapped_pe] + best_value >= self.cur_map_stat_per_pe[second_pe.name] + second_value:
                    best_value = second_value
                    best_pe = second_pe
            start_layer_idx = config.start_nodes_idx[prio - 1] - 1
            cpu = self.random.randint(0, config.num_virtual_cpu - 1)
            for i in range(start_layer_idx, start_layer_idx + len(self.app_list[prio - 1].layer_list)):
                if self.layer_list[i].is_start_node or self.layer_list[i].is_end_node:
                    self.mapping[i] = cpu
                else:
                    self.mapping[i] = best_pe.get_idx()
            prev_mapped_pe = best_pe.name
            self.cur_map_stat_per_pe[best_pe.name] += best_value

    def _init_current_map_status(self):
        self.cur_map_stat_per_pe = dict()
        for pe in self.pe_list:
            self.cur_map_stat_per_pe[pe.name] = float(0)
        return self.cur_map_stat_per_pe

    def _get_the_best(self, target_dict):
        best_value = float("inf")
        for pe in self.pe_list:
            if target_dict[pe.name] < best_value:
                best_value = target_dict[pe.name]
                best_pe = pe
        return best_value, best_pe

    def _apply_cpu_util_cst(self, mapping):
        if config.CPU_UTILIZATION != 100:
            PE.init_apps_pe_by_mapping(self.app_list, mapping, self.pe_list)
            cpu_util = CPU_utilization(self.app_list, None, self.pe_list)
            while cpu_util.constraint_function(mapping)[0] != 0:  # (penalty, util)
                for idx in range(len(mapping)):
                    l = self.layer_list[idx]
                    if not l.is_start_node and not l.is_end_node and l.pe.type == PEType.CPU:
                        best = float("inf")
                        best_pe_idx = None
                        for pe in self.pe_list:
                            target = l.time_list[pe.get_idx()]
                            if target < best:
                                best = target
                                best_pe_idx = pe.get_idx()
                        mapping[idx] = best_pe_idx

    # FIXME
    # TODO clean up
    def find_schedulable_mapping(self):
        csts = self.fitness.csts
        objs_sum = self._get_sum_fitness(self.mapping)

        target_layer = None
        best = objs_sum
        mapping = self.mapping
        ret = True
        for idx, cst in enumerate(csts):
            layer, interference = cst.get_violated_layer(mapping)
            if layer is None:
                continue

            ret = False
            # print("Violate Layer: {}".format(layer.name))
            my_best, my_mapping, _ = self._move_all_pe(layer.get_index(), mapping, best)

            lp = interference[1]
            lp_best = best
            lp_mapping = None
            if lp is not None:
                lp_best, lp_mapping, _ = self._move_all_pe(lp.get_index(), mapping, best)
                # print("After lp({}) move best: {}, dst: {}".format(lp.name, best, dst))

            hp = interference[0]
            hp_best = float("inf")
            hp_mapping = None
            if hp is not None:
                hp_best, hp_mapping = self._move_cluster(hp.get_index(), mapping, lp_best)
                # print("After hp({}) move best: {}, mapping: {}".format(hp.name, best, mapping))

            # print("(hp, lp) = ({}, {}) / best {}".format(hp_best, lp_best, best))
            candidates = [(my_best, my_mapping), (lp_best, lp_mapping), (hp_best, hp_mapping), (best, mapping)]
            local_best, mapping = reduce(lambda x, y: x if x[0] < y[0] else y, candidates)
            # print(candidates)
            # print(best, local_best, mapping)
            if local_best < best:
                self.mapping = mapping
                return False
            # if hp_best < lp_best:
            #     self.mapping = mapping
            #     # print("\tHP Mapping: {}".format(self.mapping))
            #     return False
            # elif lp_best < best:
            #     self.mapping[lp.get_index()] = lp_dst
            #     # print("\tLP Mapping: {}".format(self.mapping))
            #     return False

        return ret

    def _get_sum_fitness(self, mapping):
        values = self.fitness.calculate_fitness(mapping)
        value = reduce(lambda x, y: x + y, values)
        return value

    def _move_all_pe(self, idx, mapping, best):
        dst = -1
        new_mapping = list(mapping)
        for pe in range(self.num_pe - 1):
            new_mapping[idx] = (new_mapping[idx] + 1) % self.num_pe
            value = self._get_sum_fitness(new_mapping)
            if value < best:
                best = value
                dst = new_mapping[idx]
        return best, new_mapping, dst

    def _move_cluster(self, idx, mapping, best):
        value, _, dst = self._move_all_pe(idx, mapping, float("inf"))
        new_mapping = list(mapping)
        new_mapping[idx] = dst
        idx_list = [idx]
        while best < value:
            l = idx_list[-1] + 1
            if (l + 1) in config.start_nodes_idx or (l + 1) in config.end_nodes_idx:
                l = self.num_layer

            s = idx_list[0] - 1
            if (s + 1) in config.start_nodes_idx or (s + 1) in config.end_nodes_idx:
                s = -1

            l_value = s_value = float("inf")
            if l < self.num_layer:
                new_mapping[l], bak = dst, new_mapping[l]
                l_value = self._get_sum_fitness(new_mapping)
                new_mapping[l] = bak

            if s >= 0:
                new_mapping[s], bak = dst, new_mapping[s]
                s_value = self._get_sum_fitness(new_mapping)
                new_mapping[s] = bak

            # print("prev", new_mapping)
            if l_value > s_value:
                value = s_value
                new_mapping[s] = dst
                idx_list.insert(0, s)
            elif l_value < s_value:
                value = l_value
                new_mapping[l] = dst
                idx_list.append(l)
            else:
                break
            # print("->", new_mapping)

        # print(new_mapping, dst, best, value)
        return value, new_mapping

    def _move_large2small(self, mapping):
        large = -float("inf")
        small = float("inf")
        large_pe = None
        small_pe = None
        PE.init_apps_pe_by_mapping(self.app_list, mapping, self.pe_list)
        for idx in range(0, self.num_pe):
            if self.cur_map_stat_per_pe[self.pe_list[idx].name] > large:
                large = self.cur_map_stat_per_pe[self.pe_list[idx].name]
                large_pe = self.pe_list[idx]
            if self.cur_map_stat_per_pe[self.pe_list[idx].name] < small:
                small = self.cur_map_stat_per_pe[self.pe_list[idx].name]
                small_pe = self.pe_list[idx]

        for idx in range(0, self.num_layer):
            if mapping[idx] == large_pe.get_idx():
                mapping[idx] = small_pe.get_idx()
                self.cur_map_stat_per_pe[large_pe.name] -= self.layer_list[idx].time_list[large_pe.get_idx()]
                self.cur_map_stat_per_pe[small_pe.name] += self.layer_list[idx].time_list[small_pe.get_idx()]
                break
