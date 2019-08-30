from mapping_function import MapFunc
from fitness import Fitness
from pe import PE
import config


class ManualMapping(MapFunc):
    def __init__(self, app_list, pe_list):
        self.layer_list = [l for app in app_list for l in app.layer_list]
        self.fitness = Fitness(app_list, pe_list)
        self.app_list = app_list
        self.pe_list = pe_list

    def do_schedule(self):
        mapping = [0] * len(self.layer_list)

        # mobilenet_v2
        # mapping = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # squeezenet
        # mapping = [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.fitness.calculate_fitness(mapping)

        return [mapping]
