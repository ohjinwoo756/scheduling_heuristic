from analyzer import Analyzer
import config
from sched_simulator import SchedSimulator
from layer_graph import DirectedGraph
from demand_function import DemandFunction


class MobilityAnalyzer(Analyzer):
    instance = {}

    def __new__(cls, *args, **kwargs):
        app = args[1]
        if app in MobilityAnalyzer.instance:
            return MobilityAnalyzer.instance[app]
        inst = super(MobilityAnalyzer, cls).__new__(cls, *args, **kwargs)
        MobilityAnalyzer.instance[app] = inst
        inst.is_init = False
        return inst

    def __init__(self, app_list, app, pe_list):
        if self.is_init:
            return

        # self.app_list = app_list
        self.app_list = sorted(app_list, key=lambda _app: _app.priority)
        self.app = app
        self.pe_list = pe_list
        self.pe2layer = [[] for _ in pe_list]
        self.sched_sim = SchedSimulator([app], pe_list)
        self._max_mobility = app.get_period()
        self.is_init = True

    def _update_mobility(self):
        app = self.app
        for pe, _ in enumerate(self.pe_list):
            for t in self.pe2layer[pe]:
                demand = 0.
                for a in self.app_list:
                    if a is app:
                        break
                    # demand += self.dbf[a].get_max_demand(pe, app.get_period())
                    demand += self.dbf[a].get_max_demand(pe, self._max_mobility + t.time_list[pe])

                if demand == 0.:
                    continue

            # print("PE {}] Period {:.2f}\tMax_Mobility {:.2f}\tDemand {:.2f}".format(pe, app.get_period(), self._max_mobility, demand))
                t.set_mobility(t.mobility - demand)
                # print("\t-> {t.name:>12s}\tmobility {t.mobility:.2f}".format(t=t))

            # consider the delay from task dependency.
            for t in self.pe2layer[pe]:
                app.update_mobility(t, t.mobility)

            # consider the delay from tasks which are mapped on the same PE.
            for _pe, _ in enumerate(self.pe_list):
                if not self.pe2layer[_pe]:
                    continue
                prev_mobility = self.pe2layer[_pe][0].mobility
                for t in self.pe2layer[_pe]:
                    if t.mobility > prev_mobility:
                        t.set_mobility(prev_mobility)
                    prev_mobility = t.mobility

    def get_response_time(self):
        app = self.app
        last_layer = app.layer_list[-1]
        if app.get_priority() != 1:
            self._update_mobility()

        pe = self.mapping[last_layer.get_index()]
        _, lp_max_exec = self._get_lp_max_exec(pe, app.get_priority())

        # print("Name {app.name:>12s}\tPeriod {app.period:.2f}\tMobility {mobility:.2f} lp max_exec {lp_max_exec:.2f}".format(app=app, mobility=last_layer.mobility, lp_max_exec=lp_max_exec))
        return app.get_period() - (last_layer.mobility - lp_max_exec)

    def _get_lp_max_exec(self, pe, prio):
        lp_max_exec = 0.
        lp = None
        for a in self.app_list:
            if a.get_priority() <= prio:
                continue
            for t in a.layer_list:
                lp_pe = self.mapping[t.get_index()]
                lp_exec = t.time_list[pe]
                # print("Name {t.name:>12s}\tPriority {a.priority} > {app.priority} {lp_exec:.2f}".format(t=t, a=a, app=app, lp_exec=lp_exec))
                if pe == lp_pe and lp_max_exec < lp_exec:
                    lp_max_exec = lp_exec
                    lp = t
        return lp, lp_max_exec

    def get_schedulable_penalty(self):
        app = self.app
        last_layer = app.layer_list[-1]
        if app.get_priority() != 1:
            self._update_mobility()

        pe = self.mapping[last_layer.get_index()]
        _, lp_max_exec = self._get_lp_max_exec(pe, app.get_priority())

        return 1024 * max(-(last_layer.mobility - lp_max_exec), 0)

    def preprocess(self, mapping):
        app = self.app
        first_layer = app.layer_list[0]
        idx = first_layer.get_index()
        self.mapping = mapping
        self.dbf = {}
        for a in self.app_list:
            if a is app:
                break
            self.dbf[a] = DemandFunction(a, mapping, self.pe_list)

        self.sched_sim.do_init()
        self.sched_sim.do_simulation(mapping[idx:])  # response time of target app
        response_time = self.sched_sim.get_response_time(app)
        self.pe2layer = [[] for _ in self.pe_list]
        for layer in app.layer_list:
            self._max_mobility = max(first_layer.get_period() - response_time, 0)
            layer.set_mobility(first_layer.get_period() - response_time)
            self.pe2layer[mapping[layer.get_index()]].append(layer)

    def get_violated_layer(self):
        app = self.app

        if app.get_priority() != 1:
            self._update_mobility()

        violated_layer = None
        for layer in app.layer_list:
            pe = self.mapping[layer.get_index()]
            lp_max, lp_max_exec = self._get_lp_max_exec(pe, app.get_priority())
            if layer.mobility - lp_max_exec < 0:
                violated_layer = layer
                break

        if violated_layer is None:
            return violated_layer, [None, lp_max]

        import math
        hp_max = None
        hp_max_interference = 0.
        violated_pe = self.mapping[violated_layer.get_index()]
        for a in self.app_list:
            if a.get_priority() < app.get_priority():
                interference_list = filter(lambda l: self.mapping[l.get_index()] == violated_pe, a.layer_list)
                for l in interference_list:
                    pe = violated_pe
                    l_exec = l.time_list[pe]
                    interference = math.ceil(float(app.get_period()) / l.get_period()) * l_exec
                    # FIXME
                    if hp_max_interference < interference and (l.get_index() + 1) not in config.start_nodes_idx:
                        hp_max_interference = interference
                        hp_max = l

        return violated_layer, [hp_max, lp_max]
