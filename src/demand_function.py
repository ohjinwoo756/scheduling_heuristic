from sched_simulator import SchedSimulator
from collections import deque


class DemandFunction(object):
    instance = {}

    def __init__(self, app, mapping, pe_list):
        self.pe_exec = [0.] * len(pe_list)
        self.period = app.get_period()

        first_layer = app.layer_list[0]
        idx = first_layer.get_index()
        self.sched_sim = SchedSimulator([app], pe_list)
        self.sched_sim.do_init()
        self.sched = self.sched_sim.do_simulation(mapping[idx:])  # response time of target app

    def __new__(cls, *args, **kwargs):
        app = args[1]
        if app in DemandFunction.instance:
            return DemandFunction.instance[app]
        inst = super(DemandFunction, cls).__new__(cls, *args, **kwargs)
        DemandFunction.instance[app] = inst
        return inst


class DemandFunctionRT(DemandFunction):
    def __init__(self, app_list, app, mapping, pe_list):
        super(DemandFunctionRT, self).__init__(app, mapping, pe_list)
        for idx, pe_time in enumerate(self.sched.busy_time):
            for time in pe_time:
                self.pe_exec[idx] += time[1] - time[0]

    def get_max_demand(self, pe, t):
        import math
        rest_demand = min(t % self.period, self.pe_exec[pe]) if t % self.period != 0 else 0
        return math.floor(t / self.period) * self.pe_exec[pe] + rest_demand


class DemandFunctionMobility(DemandFunction):
    def __init__(self, app_list, app, mapping, pe_list):
        super(DemandFunctionMobility, self).__init__(app, mapping, pe_list)
        self.app = app
        self.app_list = app_list
        self.mapping = mapping

        response_time = self.sched_sim.get_response_time(app)
        first_layer = app.layer_list[0]
        max_mobility = first_layer.get_period() - response_time

        self.busy_time = []
        self.delay_time = []

        self.pe2max_low_inter = []
        for pe, _ in enumerate(pe_list):
            lp_max_exec = self._get_lp_max_exec(pe, app.get_priority())
            self.pe2max_low_inter.append(lp_max_exec)

        self.busy_exec = [0. for _ in pe_list]
        for pe, pe_time in enumerate(self.sched.sched_time):
            self.busy_time.append(deque(pe_time))
            self.delay_time.append([max_mobility - max(s.mobility - self.pe2max_low_inter[pe], 0) for s in self.sched.sched[pe].keys()])
            for time in pe_time:
                self.pe_exec[pe] += time[1] - time[0]
                if self.busy_exec[pe] < time[1] - time[0]:
                    self.busy_exec[pe] = time[1] - time[0]

    def _get_lp_max_exec(self, pe, prio):
        lp_max_exec = 0.
        app = self.app
        for a in self.app_list:
            if a.get_priority() <= prio:
                continue
            for t in a.layer_list:
                lp_pe = self.mapping[t.get_index()]
                lp_exec = t.time_list[pe]
                # print("Name {t.name:>12s}\tPriority {a.priority} > {app.priority} {lp_exec:.2f}".format(t=t, a=a, app=app, lp_exec=lp_exec))
                if pe == lp_pe and lp_max_exec < lp_exec:
                    lp_max_exec = lp_exec
        return lp_max_exec

    def _get_rest_demand(self, t, busy_time, delay_time):
        max_demand = 0.
        _busy_time = deque(busy_time)
        # print(" [ start t={} {} ] ".format(t, self.period))
        flags = [True for _ in busy_time] + [False]
        base_idx = 0
        for idx, _ in enumerate(busy_time):
            demand = 0.
            prev_end = _busy_time[0][0]
            _t = t

            for idx2, _ in enumerate(busy_time):
                _busy_time.append((_busy_time[idx2][0] + self.period, _busy_time[idx2][1] + self.period))
            # calculate the demand of a window starting from prev_end
            for idx2, (time, f) in enumerate(zip(_busy_time, flags)):
                # print(prev_end, time[0])
                flag = base_idx + idx2 < len(delay_time) and f
                if prev_end < time[0] + (delay_time[base_idx + idx2] if flag else 0):
                    _t -= time[0] - prev_end
                    if _t < 0:
                        break

                slot = min(time[1] - time[0], _t)
                # slot = time[1] - time[0]
                demand += slot
                if slot >= _t:
                    break
                else:
                    _t -= slot
                    prev_end = time[1] + (delay_time[base_idx + idx2] if flag else 0)
            for _ in busy_time:
                del _busy_time[-1]
            assert len(busy_time) == len(_busy_time)

            # print("1.", demand, max_demand)
            if max_demand < demand:
                max_demand = demand
            # print("2.", demand, max_demand)
            _busy_time.rotate(-1)
            _busy_time[-1] = (_busy_time[-1][0] + self.period, _busy_time[-1][1] + self.period)
            base_idx += 1
            flags[-base_idx] = False
        # print(" [ end {} ] ".format(max_demand))

        return max_demand

    def get_max_demand(self, pe, t):
        if t <= 0:
            return 0

        import math
        rest_demand = self._get_rest_demand(t % self.period, self.busy_time[pe], self.delay_time[pe]) if t % self.period != 0 else 0
        return math.floor(t / self.period) * self.pe_exec[pe] + rest_demand

    def get_max_busy_exec(self, pe):
        return self.busy_exec[pe]

    def get_max_demand_partial_tasks(self, pe, t, task_list):
        busy_time = deque()
        time_info = self.sched.sched[pe]
        pe_exec = 0.
        for time in time_info:
            if time[0] not in task_list:
                continue
            busy_time.append((time[1], time[2] + time[3]))
            pe_exec += time[2] + time[3] - time[1]

        import math
        rest_demand = self._get_rest_demand(t % self.period, busy_time) if t % self.period != 0 else 0
        return math.floor(t / self.period) * pe_exec + rest_demand
