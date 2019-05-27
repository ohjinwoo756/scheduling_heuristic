
class Task():

    def __init__(self, index, net_l):
        self.index = index
        self.name = str(net_l.name)
        self.type = str(net_l.type)
        self.parents = []
        self.childs = []

    def set_processor_profile(self, prof_l):
        self.cpu = prof_l.cpu
        self.gpu = prof_l.gpu
        self.npu = prof_l.npu

