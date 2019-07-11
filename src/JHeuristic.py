from mapping_function import MapFunc
from fitness import Fitness
# TODO: from cpu_utilization import CPU_utilization
import config

class JHeuristic(MapFunc):

    def __init__(self, app_list, pe_list):
        self.app_list = app_list
        self.num_app = len(app_list)
        self.layer_list = [l for app in app_list for l in app.layer_list]
        self.num_layer = len(self.layer_list)
        self.num_pe = len(pe_list)
        self.fitness = Fitness(app_list, pe_list)

        # Construct OCT (Optimal Cost Table)
        self.construct_OCT()

    def do_schedule(self):
        # Task prioritizing phase
        self.prioritize_tasks()

        # Processor selection phase
        self.assign_task_to_processor()

        mapping = [0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2, 0, 0, 0, 0]
        return [mapping]

    def construct_OCT(self):
        pass
        # print list(self.network.layer)
        # print type(str(self.network.layer[0].name))
        # tmp = str(self.network.layer[0].name)
        # print type(self.profile.layer[0].name)
        # print self.profile.layer[0].name[1]
        # print self.network.layer
        # net_len = len(self.network.layer)
        # print "network length: %d" & net_len

    def prioritize_tasks(self):
        pass

    def assign_task_to_processor(self):
        pass
