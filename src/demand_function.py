from sched_simulator import SchedSimulator
from collections import deque


class DemandFunction(object):
    instance = {}

    def __new__(cls, *args, **kwargs):
        app = args[0]
        if app in DemandFunction.instance:
            return DemandFunction.instance[app]
        inst = super(DemandFunction, cls).__new__(cls, *args, **kwargs)
        DemandFunction.instance[app] = inst
        return inst

    def __init__(self, app, mapping, pe_list):
        self.pe_exec = [0.] * len(pe_list)
        self.period = app.get_period()

        first_layer = app.layer_list[0]
        idx = first_layer.get_index()
        sched_sim = SchedSimulator([app], pe_list)
        sched_sim.do_init()
        self.sched, _ = sched_sim.do_simulation(mapping[idx:])  # response time of target app
        self.busy_time = []
        self.busy_exec = [0. for _ in pe_list]
        for idx, pe_time in enumerate(self.sched.busy_time):
            self.busy_time.append(deque(pe_time))
            for time in pe_time:
                self.pe_exec[idx] += time[1] - time[0]
                if self.busy_exec[idx] < time[1] - time[0]:
                    self.busy_exec[idx] = time[1] - time[0]

    def _get_rest_demand(self, t, busy_time):
        max_demand = 0.
        _busy_time = deque(busy_time)
        # print(" [ start t={}] ".format(t))
        for idx, _ in enumerate(busy_time):
            demand = 0.
            prev_end = busy_time[0][0]
            # print(busy_time)
            _t = t
            for time in busy_time:
                # print(prev_end, time[0])
                if prev_end < time[0]:
                    _t -= time[0] - prev_end
                    if _t < 0:
                        break

                slot = time[1] - time[0]
                demand += slot
                if slot >= _t:
                    break
                else:
                    _t -= slot
                    prev_end = time[1]
            # print("1.", demand, max_demand)
            if max_demand < demand:
                max_demand = demand
            # print("2.", demand, max_demand)
            _busy_time.rotate(-1)
            busy_time[-1] = (busy_time[-1][0] + self.period, busy_time[-1][1] + self.period)
        # print(" [ end {} ] ".format(max_demand))

        return max_demand

    def get_max_demand(self, pe, t):
        import math
        rest_demand = self._get_rest_demand(t % self.period, self.busy_time[pe]) if t % self.period != 0 else 0
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
