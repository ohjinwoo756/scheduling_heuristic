from application import Application

class PEFT():

    def __init__(self, network, profile):
        self.target_apps = []
        for idx in range(0, len(network)): # iterates as the total number of networks
            self.target_apps.append(Application(network[idx], profile[idx]))

        # Construct OCT (Optimal Cost Table)
        self.construct_OCT()

    def transform_to_tasks(self, network, profile):
        pass

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

    def do_schedule(self):
        # Task prioritizing phase
        self.prioritize_tasks()

        # Processor selection phase
        self.assign_task_to_processor()

    def prioritize_tasks(self):
        pass

    def assign_task_to_processor(self):
        pass
