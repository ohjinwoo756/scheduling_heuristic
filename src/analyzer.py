class Analyzer(object):
    def __init__(self, app_list, app, pe_array):
        pass

    def get_response_time(self, app):
        raise NotImplementedError

    def get_schedulable_penalty(self, app):
        raise NotImplementedError

    def preprocess(self, mapping):
        raise NotImplementedError
