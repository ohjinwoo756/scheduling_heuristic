import config
from objective import Objective
from sched_simulator import SchedSimulator


class Response_time(Objective):
    def __init__(self, app_list, app, pe_list):
        super(Response_time, self).__init__(app_list, app, pe_list)
        analyzer_module = analyzer_class = None
        if config.analyzer == "rt":
            analyzer_module = __import__("rt_analyzer")
            analyzer_class = getattr(analyzer_module, "RTAnalyzer")
            # self.sched_sim = SchedSimulator(app_list, pe_list)
        elif config.analyzer == "mobility":
            analyzer_module = __import__("mobility_analyzer")
            analyzer_class = getattr(analyzer_module, "MobilityAnalyzer")
        self.analyzer = analyzer_class(app_list, app, pe_list)

    def _objective_function(self, mapping):
        self.analyzer.preprocess(mapping)
        return self.analyzer.get_response_time(),
        # self.sched_sim.do_init()
        # self.sched_sim.do_simulation(self.mapping)  # response time of target app
        # return self.sched_sim.get_response_time(self.app),
