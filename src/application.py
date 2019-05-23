from task import Task

class Application():

    def __init__(self, network, profile):
        # Program input
        self.network = network
        self.profile = profile
        self.layer_list = list(network.layer)
        self.profile_list = list(profile.layer)
        self.network_len = len(self.layer_list)

        # Variables from input to real usage
        self.tasks = []
        self.name2task_dict = {} # hash table for speed up
        self.make_tasks() # Input => Tasks

    def make_tasks(self):
        self.make_initial_tasks() # Except for layer precedence
        self.apply_precedences()

    def make_initial_tasks(self):
        # Example of duplicate for loop
        # self.layer_list = [l for app in app_list for l in app.layer_list]

        # 1. From network input
        for index, net_l in enumerate(self.layer_list):
            self.tasks.append(Task(index, net_l)) # Make 'Task' object
            self.name2task_dict[self.tasks[index].name] = self.tasks[index] # register to dictionary

        # 2. From profile input
        for index, prof_l in enumerate(self.profile_list):
            self.tasks[index].cpu = prof_l.cpu
            self.tasks[index].gpu = prof_l.gpu
            self.tasks[index].npu = prof_l.npu

    def apply_precedences(self):
        for l_idx, net_l in enumerate(self.layer_list):
            target_parents_name = []
            for idx, parent in enumerate(list(net_l.bottom)):
                target_parents_name.append(str(parent)) # convert to string
                parent_obj = self.name2task_dict[target_parents_name[idx]]
                self.tasks[l_idx].parents.append(parent_obj) # add parent info
                parent_obj.childs.append(self.tasks[l_idx]) # add chind info

        # debug
        # for l_idx, net_l in enumerate(self.layer_list):
        #     for idx in range(0, len(net_l.top)):
        #         p1 = self.tasks[l_idx].childs[idx]
        #         p2 = self.tasks[p1.index]
        #         print "Child matched? ", p1 == p2
        #     for idx in range(0, len(net_l.bottom)):
        #         t1 = self.tasks[l_idx].parents[idx]
        #         t2 = self.tasks[t1.index]
        #         print "Parent matched? ", t1 == t2
