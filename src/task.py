
class Task():

    def __init__(self):
        self.name = None
        self.type = None
        self.parent = []
        self.child = []

    def set_name(self, layer_name):
        self.name = layer_name

    def set_type(self, layer_type):
        self.type = layer_type
