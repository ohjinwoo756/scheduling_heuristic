import config
from constraint import Constraint


class Deadline(Constraint):
    def __init__(self, app_list, app, pe_list):
        super(Deadline, self).__init__(app_list, app, pe_list)
        analyzer_module = analyzer_class = None
        if config.analyzer == "rt":
            analyzer_module = __import__("rt_analyzer")
            analyzer_class = getattr(analyzer_module, "RTAnalyzer")
        elif config.analyzer == "mobility":
            analyzer_module = __import__("mobility_analyzer")
            analyzer_class = getattr(analyzer_module, "MobilityAnalyzer")
        self.analyzer = analyzer_class(app_list, app, pe_list)

    def _constraint_function(self, mapping):
        self.analyzer.preprocess(mapping)
        penalty = self.analyzer.get_schedulable_penalty()
        return penalty,

    def get_violated_layer(self, mapping):
        self.analyzer.preprocess(mapping)
        layer = self.analyzer.get_violated_layer()
        return layer
