class Constraint(object):
    def __init__(self, app_list, app, pe_array):
        self.app_list = app_list
        self.app = app
        self.pe_array = pe_array
        self.mapping = None
        self.gene2cst = {}

    def mapping2hashkey(self, mapping):
        return ''.join(str(pe) for pe in mapping)

    def check_hash(self, mapping):
        value = self.mapping2hashkey(mapping)
        if value in self.gene2cst:
            return self.gene2cst[value]
        else:
            return None

    def save_hash(self, mapping, value):
        assert value not in self.gene2cst
        self.gene2cst[self.mapping2hashkey(mapping)] = value

    def _constraint_function(self, mapping):
        raise NotImplementedError

    def constraint_function(self, mapping):
        value = self.check_hash(mapping)
        if value is not None:
            return value

        value = self._constraint_function(mapping)
        self.save_hash(mapping, value)
        return value
