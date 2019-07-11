from mapping_function import MapFunc
from fitness import Fitness


class Heuristic(MapFunc):
    def __init__(self, app_list, pe_list):
        self.app_list = app_list
        self.pe_list = pe_list
        self.layer_list = [l for app in app_list for l in app.layer_list]
        self.num_layer = len(self.layer_list)
        self.num_pe = len(pe_list)
        self.fitness = Fitness(app_list, pe_list)

    def do_schedule(self):
        self.mapping = [0] * len(self.layer_list)
        self._set_initial_mapping()
        if self._pass_constraint():
            self.fitness.calculate_fitness(self.mapping)
            return [self.mapping]
        else:
            return []  # no solution

    def _set_initial_mapping(self, mapping):
        raise NotImplementedError

    def _pass_constraint(self, mapping):
        raise NotImplementedError
