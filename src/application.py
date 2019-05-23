from task import Task

class Application():

    def __init__(self, network, profile):
        self.network = network
        self.profile = profile
        self.layer_list = list(network.layer)
        self.profile_list = list(profile.layer)
        self.network_len = len(self.layer_list)
        self.tasks = [] # Parsed inputs (network, profile)

        self.parse_layers_to_tasks()

    def parse_layers_to_tasks(self):
        # Parse network information
        for idx, net_l in enumerate(self.layer_list):
            self.tasks.append(Task()) # Make 'Task' object
            self.tasks[idx].set_name(net_l.name)
            self.tasks[idx].set_type(net_l.type)

        # Parse network profile
        for idx, prof_l in enumerate(self.profile_list):
            pass

        # Example
        # self.layer_list = [l for app in app_list for l in app.layer_list]
        # print self.tasks[0].name
