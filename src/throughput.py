import config
from objective import Objective
from sched_simulator import SchedSimulator


class Throughput(Objective):
    def __init__(self, app_list, app, pe_array):
        super(Throughput, self).__init__(app_list, app, pe_array)
        self.thresh = 100
        self.sched_sim = SchedSimulator(app_list, pe_array)

    def _get_converged_period(self, end_iteration):
        max_interval = 0.
        sim = self.sched_sim
        for pe_idx in range(config.num_virtual_cpu):
            if sim.pe_end_time[pe_idx][0] == -1:
                continue
            interval = sim.pe_end_time[pe_idx][end_iteration - 1] - sim.pe_start_time[pe_idx]
            if max_interval < interval:
                max_interval = interval
        return max_interval / end_iteration

    def _objective_function(self, mapping):
        sim = self.sched_sim
        sim.do_init()
        start = 0
        end = 50

        prev_throughput = 0
        prev_delta = 0
        while True:
            sim.do_simulation(mapping, (start, end))

            # Simulation stopping condition
            throughput = self._get_converged_period(end)
            delta = abs(throughput - prev_throughput)
            d_delta = abs(prev_delta - delta)
            # thresh: 100(ms)
            if (delta < self.thresh and d_delta < self.thresh):
                break
            prev_delta = delta
            prev_throughput = throughput

            start = end
            end += 5

        return self._get_converged_period(end),
