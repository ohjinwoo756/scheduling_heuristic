from operator import add
from Layer import Layer

class MergedLayer(Layer):
    def __init__(self, layer_list):
        name_list = map(lambda n: n.get_name(), layer_list)
        self.name = reduce(lambda n1, n2: n1 + "/" + n2, name_list)
        del name_list

        first_layer = layer_list[0]
        self.period = first_layer.get_period()

        layer_times = map(lambda n: n.time_list, layer_list)
        # print(len(layer_times))
        # for t in layer_times:
        #     print(t)
        self.time_list = reduce(lambda t1, t2: map(add, t1, t2),layer_times)
        del layer_times
        # print(self.time_list)

        self.app = first_layer.get_app()
        self.index = first_layer.get_index()
        self.priority = -1
        self.size = first_layer.size

        self.transfer_time = 0.
        for l in layer_list:
            self.transfer_time += self.app.graph.get_transition_time(l)
