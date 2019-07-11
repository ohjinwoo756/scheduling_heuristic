class Objective(object):
    def __init__(self, app_list, app, pe_array):
        self.app_list = app_list
        self.app = app
        self.pe_array = pe_array
        self.mapping = None
        self.gene2obj = {}

    def mapping2hashkey(self, mapping):
        return ''.join(str(pe) for pe in mapping)

    def check_hash(self, mapping):
        value = self.mapping2hashkey(mapping)
        if value in self.gene2obj:
            return self.gene2obj[value]
        else:
            return None

    def save_hash(self, mapping, value):
        assert value not in self.gene2obj
        self.gene2obj[self.mapping2hashkey(mapping)] = value

    def _objective_function(self):
        raise NotImplementedError

    def objective_function(self, mapping):
        value = self.check_hash(mapping)
        if value is not None:
            return value

        value = self._objective_function(mapping)
        self.save_hash(mapping, value)
        return value
