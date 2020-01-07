from random import SystemRandom
import numpy as np

from deap import base
from deap import creator
from deap import tools

from mapping_function import MapFunc
from sched_simulator import SchedSimulator
from fitness import Fitness
import config


class GA(MapFunc):

    def __init__(self, app_list, pe_list):
        self.random = SystemRandom()
        self.app_list = app_list
        self.num_app = len(app_list)
        self.layer_list = [l for app in app_list for l in app.layer_list]
        self.num_layer = len(self.layer_list)
        self.num_pe = len(pe_list)
        # self.sched_sim = SchedSimulator(app_list, pe_list, cpu_core_distribution)
        self.fitness = Fitness(app_list, pe_list)

        # variables for chromosome initialization with clustering
        self._cluster_size = self.num_layer / self.num_pe
        self.cluster_size = self._cluster_size
        self.cluster_pe = self.random.randint(0, self.num_pe - 1)
        self.cluster_iterator = 1
        self.individual_index = 0
        self.app_index = 0

        # for debug
        self.divide_iter = 0.5

        self.inc_iter = 1

        self.population = 64
        self.population_index = 0

        # Optimum fitness value is negative
        creator.create("Fitness", base.Fitness, weights=(-1.0, ) * len(self.fitness.objs))
        creator.create("Individual", list, fitness=creator.Fitness)

        self.toolbox = base.Toolbox()
        # XXX for using multi processors
        self.toolbox.register("generate_processor", self.generate_processor)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.generate_processor, self.num_layer)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxUniform, indpb=0.1)
        upList = [self.num_pe - 1] * self.num_layer
        for start, end in zip(config.start_nodes_idx, config.end_nodes_idx):
            upList[start - 1] = config.num_virtual_cpu - 1  # The first layer: only CPU
            upList[end - 1] = config.num_virtual_cpu - 1  # The last layer: only CPU
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=upList, indpb=4 * len(app_list) / float(self.num_layer))
        # self.toolbox.register("select", tools.selTournament, k=self.population, tournsize=4)
        # self.toolbox.register("select", tools.selNSGA2, k=8)
        self.toolbox.register("select", tools.selSPEA2, k=16)
        # self.toolbox.register("select_best", tools.selBest, k=self.population / 16)
        self.toolbox.register("select_random", tools.selRandom, k=self.population)

        # self.toolbox.register("evaluate", self.sched_sim.do_simulation)
        self.toolbox.register("evaluate", self.fitness.calculate_fitness)

    def _init_cluster(self, for_next_gene):
        self.cluster_size = self.random.randint(self._cluster_size * 3 / 4, self._cluster_size * 5 / 4)
        self.cluster_iterator = 1
        if for_next_gene:
            self.cluster_pe = self.random.randint(0, self.num_pe - 1)
        else:
            self.cluster_pe = (self.random.randint(1, self.num_pe - 1) + self.cluster_pe) % self.num_pe

    def generate_processor(self):
        self.individual_index += 1

        if self.individual_index in config.start_nodes_idx or self.individual_index in config.end_nodes_idx:
            if self.individual_index == self.num_layer:
                if self.population_index % 8 == 1:
                    self._init_cluster(True)
                self.individual_index = 0
                self.app_index = 0
                self.population_index += 1
            if self.individual_index in config.start_nodes_idx:
                self.app_index += 1
            pe = self.random.randint(0, config.num_virtual_cpu - 1)
            return pe  # only CPU

        ret = -1
        if self.population_index % 8 == 1:
            if self.cluster_iterator <= self.cluster_size:
                self.cluster_iterator += 1
            else:
                self._init_cluster(False)
            ret = self.cluster_pe
        elif self.population_index % 8 == 2:
            ret = self.num_pe - 1  # FIXME npu
        elif self.population_index % 16 == 3:
            ret = self.num_pe - self.app_index
        elif self.population_index % 16 == 11:
            ret = (config.num_virtual_cpu + self.app_index - 1) % self.num_pe
        else:
            ret = self.random.randint(0, self.num_pe - 1)

        return ret

    def do_schedule(self):
        mate_prob = 0.7
        mutate_prob = 0.3

        pop = self.toolbox.population(n=self.population)
        app_idx = 0
        for idx in range(len(pop[0])):
            if (idx + 1) in config.start_nodes_idx:
                app_idx += 1
            else:
                pop[4][idx] = (config.num_virtual_cpu + app_idx - 1) % self.num_pe
                pop[5][idx] = self.num_pe - app_idx

        hof = tools.ParetoFront()  # hall of fame

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        self.eaSimpleWithLocalSearch(pop, mate_prob, mutate_prob, stats, hof)

        return hof

    def varAnd(self, population, toolbox, cxpb, mutpb):
        offspring = [toolbox.clone(ind) for ind in population]

        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if self.random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if self.random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        return offspring

    def calc_delta(self, sig_idx, ind, obj_idx):
        def get_cost(value):
            if len(ind.fitness.values) != self.num_app and obj_idx >= self.num_app:
                cost = value[obj_idx]
            else:
                cost = reduce(lambda x, y: x + y, value)
            return cost

        prev_value = ind.fitness.values
        prev_cost = get_cost(prev_value)
        prev_processor_index = ind[sig_idx]

        num_virtual_cpu = config.num_virtual_cpu

        if (sig_idx + 1) in config.start_nodes_idx and ind[sig_idx + 1] >= num_virtual_cpu:
            return False

        if (sig_idx + 1) in config.end_nodes_idx and ind[sig_idx - 1] >= num_virtual_cpu:
            return False

        if sig_idx > 0 and sig_idx < self.num_layer - 1 and ind[sig_idx] != ind[sig_idx - 1] and ind[sig_idx] != ind[sig_idx + 1]:
            ind[sig_idx] = ind[sig_idx - 1]
            next_value_1 = self.toolbox.evaluate(ind)
            next_cost_1 = get_cost(next_value_1)

            ind[sig_idx] = ind[sig_idx + 1]
            next_value_2 = self.toolbox.evaluate(ind)
            next_cost_2 = get_cost(next_value_2)
            if next_cost_1 < next_cost_2:
                next_value, next_cost = next_value_1, next_cost_1
                ind[sig_idx] = ind[sig_idx - 1]
            else:
                next_value, next_cost = next_value_2, next_cost_2
                ind[sig_idx] = ind[sig_idx + 1]
        elif sig_idx > 0 and ind[sig_idx] != ind[sig_idx - 1]:
            ind[sig_idx] = ind[sig_idx - 1]
            next_value = self.toolbox.evaluate(ind)
            next_cost = get_cost(next_value)
        elif sig_idx < self.num_layer - 1 and ind[sig_idx] != ind[sig_idx + 1]:
            ind[sig_idx] = ind[sig_idx + 1]
            next_value = self.toolbox.evaluate(ind)
            next_cost = get_cost(next_value)
        else:
            return False

        # next_cost = next_value[obj_idx]

        result = prev_cost - next_cost
        if result > 0:
            ind.fitness.values = next_value
            return True
        else:
            ind[sig_idx] = prev_processor_index
            return False

    def one_bit_flip_local_optimization(self, invalid_ind):
        obj_idx = self.random.randint(0, len(invalid_ind[0].fitness.values) - 1)
        for ind in invalid_ind:
            sigma = range(0, self.num_layer)
            improved = True
            while improved:
                improved = False
                for sig_idx in sigma:
                    if self.calc_delta(sig_idx, ind, obj_idx):
                        improved = True

    def shallow_local_optimization(self, invalid_ind):
        obj_idx = self.random.randint(0, len(invalid_ind[0].fitness.values) - 1)
        for ind in invalid_ind:
            sigma = range(0, self.num_layer)
            for sig_idx in sigma:
                if self.calc_delta(sig_idx, ind, obj_idx):
                    break

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

    def find_schedulable_mapping(self, invalid_ind):
        csts = self.fitness.csts
        # print("before", invalid_ind[0])
        for ind in invalid_ind:
            mapping = ind
            objs_sum = self._get_sum_fitness(mapping)

            target_layer = None
            best = objs_sum
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
                    ind[:] = mapping
        # print("after ", invalid_ind[0])

    def eaSimpleWithLocalSearch(self, population, cxpb, mutpb, stats=None, halloffame=None, verbose=__debug__):
        toolbox = self.toolbox
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        invalid_ind = [ind for ind in population if not ind.fitness.valid]

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update hall of fame
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream

        # Start GA process
        gen = 1
        change_point = 0  # Point where optimum fitness value changed.
        prev_mins = [float("inf")] * len(self.fitness.objs)
        while (True):
            offspring = toolbox.select(population)

            offspring = self.varAnd(offspring, toolbox, cxpb, mutpb)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # self.find_schedulable_mapping(offspring)
            if gen - change_point > 32 * self.inc_iter:
                self.one_bit_flip_local_optimization(offspring)
            elif self.random.random() < 0.25:
                self.shallow_local_optimization(offspring)

            if gen - change_point > 8 * self.inc_iter:
                mutpb = max(mutpb + 0.1, 0.8)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # population[:] = toolbox.select_pop_1(population + offspring) + toolbox.select_pop_2(offspring)  # Replacement
            population[:] = toolbox.select_random(population + offspring)  # Replacement
            # population[:] = offspring  # Replacement

            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print logbook.stream

            for idx, prev_min in enumerate(prev_mins):
                if prev_min > record['min'][idx]:
                    prev_mins[idx] = record['min'][idx]
                    change_point = gen
                    mutpb = 0.3
            if gen - change_point > 64 * self.inc_iter:
                break
            gen = gen + 1

        return population, logbook
