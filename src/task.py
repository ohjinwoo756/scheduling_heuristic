
class Task():

    def __init__(self, index, net_l):
        self.index = index
        self.name = str(net_l.name)
        self.type = str(net_l.type)
        self.cpu = None
        self.gpu = None
        self.npu = None
        self.parents = []
        self.childs = []

