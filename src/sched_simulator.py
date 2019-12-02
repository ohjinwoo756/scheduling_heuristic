from priority_queue import PriorityQueue
from sortedcontainers import SortedSet
from GanttChart import GanttChart
from app import Application

import config  # Utilization variable shared across modules.


class Schedule(object):
    def __init__(self, num_pe):
        self.sched = [[] for _ in range(num_pe)]
        self.sched_time = [[] for _ in range(num_pe)]
        self.busy_time = [[] for _ in range(num_pe)]

    def add_sched(self, time_tuple):
        pe, l, t, end_time, transfer_time = time_tuple
        self.sched[pe].append((l, t, end_time, transfer_time))
        self.sched_time[pe].append((t, end_time))
        busy_time = self.busy_time[pe]
        if busy_time and busy_time[-1] and busy_time[-1][1] == t:
            busy_time[-1] = (busy_time[-1][0], end_time + transfer_time)
        else:
            busy_time.append((t, end_time + transfer_time))

    def print_sched(self):
        for idx, pe_time in enumerate(self.sched):
            for time in pe_time:
                print(idx, time)

    def print_busy_sched(self):
        for idx, pe_time in enumerate(self.busy_time):
            for time in pe_time:
                print(idx, time)


class SchedSimulator(object):
    def __init__(self, app_list, pe_list):
        self.app_list = app_list
        # self.app_list = sorted(app_list, key=lambda _app: _app.priority)
        self.layer_list = [l for _app in app_list for l in _app.layer_list]
        self.layer_set = set(self.layer_list)
        self.gene2fit = {}

        # FIXME
        self.single_mode = False
        first_node = app_list[0].layer_list[0]
        if len(app_list) == 1 and first_node.get_index() != 0:
            self.single_mode = True

        self.num_layer = len(self.layer_list)
        self.throughput_thresh = 100

        self.draw_iteration = 1

        self.num_pe = len(pe_list)

        self.prio_step = len(self.layer_list)  # Total number of layers
        self.pe_list = pe_list

        # variables for CPU utilization constraint
        self.elapsed_time_per_pe = [0] * self.num_pe

        self._ready_queues = [PriorityQueue() for _ in range(self.num_pe)]
        self._rq_set = set()

    def _init_ready_queue(self):
        self._ready_queues = [PriorityQueue() for _ in range(self.num_pe)]
        self._rq_set = set()

    def _init_all_apps(self):
        # application initialization
        #   => Edge, Layer initialization
        for app in self.app_list:
            app.do_init()

    def do_init(self):
        self.iteration = [0] * self.num_layer
        self.response_time = [0.0] * len(self.app_list)

        self.pe_start_time = [-1 for _ in range(self.num_pe)]
        self.pe_end_time = [[-1] for _ in range(self.num_pe)]

        self._init_ready_queue()
        self._init_all_apps()

        self.timeline = SortedSet()  # FIXME need to change name. (next_sim_time?)
        offset_set = set()
        for l in self.layer_list:
            if l.offset >= 0:
                offset_set.add(l.offset)
        self.timeline.update(list(offset_set))
        self.occupy_times = [0] * self.num_pe


    def find_runnable_layers(self, t):
        runnable_layers = []
        layer_list = self.layer_set - self._rq_set
        for app in self.app_list:
            for l in app.layer_list:
                if l.need_in_edge_check and l.need_out_edge_check \
                        and l.offset <= t and app.check_runnable(l, t):
                    runnable_layers.append(l)

        return runnable_layers

    def _enqueue(self, t):
        runnable_layers = self.find_runnable_layers(t)
        for l in runnable_layers:
            if self.single_mode:
                idx = l.get_app_index()
            else:
                idx = l.get_index()
            pe = l.pe.get_idx()
            # prio = l.get_priority() + l.iteration * self.prio_step
            prio = l.get_app_priority() * self.prio_step + l.iteration
            # print("Queue " + str(pe) + " insert : " + l.name + " prio : " + str(prio))
            self._ready_queues[pe].insert(prio, l)
            self._rq_set.add(l)

    def _set_pe_time(self, pe, iteration, start_time, end_time):
        if self.pe_start_time[pe] == -1:
            self.pe_start_time[pe] = start_time
        try:
            if self.pe_end_time[pe][iteration] < end_time:
                self.pe_end_time[pe][iteration] = end_time
        except IndexError:
            assert iteration == len(self.pe_end_time[pe])
            self.pe_end_time[pe].append(end_time)

    @staticmethod
    def _get_csts_and_objs(fitness, mapping):
        objs = []
        csts = []
        for idx, cst in enumerate(fitness.csts):
            if cst is not None:
                cst_value = cst.constraint_function(mapping)
                csts.append(cst_value)
            else:
                csts.append((0,))

        for idx, obj in enumerate(fitness.objs):
            obj_value = obj.objective_function(mapping)
            objs.append(obj_value)
        return csts, objs

    def _draw_gantt(self, gantt, gantt_name, mapping, fitness):
        csts, objs = SchedSimulator._get_csts_and_objs(fitness, mapping)
        available_results = True
        for idx, (cst, value) in enumerate(zip(fitness.csts, csts)):
            if value[0] != 0:
                available_results = False
                break
        config.available_results = available_results

        if available_results:
            print("\nPE Mapping per layer: " + str(mapping))

            if len(objs) > config.num_of_app:
                print("\n\t[ Whole Objective ]")
            for idx, value in enumerate(objs):
                if idx >= config.num_of_app:
                    print("\t\tObjective function value [by Energy Consumption] :\t %d" % value[0])

            if len(csts) > config.num_of_app:
                print("\n\t[ Whole Constraint ]")
            for idx, (cst, value) in enumerate(zip(fitness.csts, csts)):
                if idx >= config.num_of_app:
                    print("\t\tConstraint function value [by %s] :\t %.2f -> %.2f" % (type(cst).__name__, value[-1], value[0]))

            objs_result = []
            for idx, app in enumerate(self.app_list):
                print("\n\t[ %s (Period: %d, Priority: %d) ]" % (app.name, app.get_period(), app.get_priority()))
                print("\t\tObjective function value [by %s]:\t%.2f" % (config.app_to_obj_dict[idx], objs[idx][0]))
                if config.app_to_cst_dict[idx] != 'None':
                    print("\t\tConstraint function value [by %s]:\t%.2f" % (config.app_to_cst_dict[idx], csts[idx][-1]))
                config.objs_result_by_app[idx].append(round(objs[idx][0], 2))
                objs_result.append(round(objs[idx][0], 2))

            # XXX: for short file name
            config.file_name = "{}{}_{}_{}_{}_{}".format(config.save_path + "/" + "#{}_".format(config.gantt_chart_idx) + config.name, str(config.sched_method), str(config.processor), str(config.period), str(config.cpu_config), str(objs_result))
            # config.file_name = "{}{}_{}_{}_{}_{}_{}_{}_{}".format(config.save_path + "/" + config.name, str(config.sched_method), str(config.processor), str(config.priority), str(config.period), str(config.cpu_config), str(config.objs), str(objs_result), str(config.csts))
            gantt.file_name = config.file_name + ".png"
            gantt.draw_gantt_chart()

    def _pop_and_get_layer_info(self, q):
        _, l = q.pop()  # pop layer from _ready_queues
        self._rq_set.remove(l)
        if self.single_mode:
            layer_idx = l.get_app_index()
        else:
            layer_idx = l.get_index()
        pe = l.pe.get_idx()
        app = l.get_app()
        return l, layer_idx, pe, app

    def _update_timeline(self, l, time):
        timeline = self.timeline
        if l.offset >= 0:
            # print("ID: {} Name: {} Time: {} {} update".format(id(l), l.name, l.offset, l.offset + l.get_period()))
            # timeline.add(l.offset)
            # timeline.add((l.iteration + 1) * l.get_period())
            timeline.add(l.get_period() + l.offset)
            l.set_offset(l.get_period() + l.offset)
            # print l.get_period()
        timeline.add(time)

    def do_simulation(self, mapping, iterations=(0, 1), draw_gantt=False, gantt_name="test.png", fitness=None):
        if draw_gantt:
            pe_names = [pe.name for pe in self.pe_list]
            gantt = GanttChart(gantt_name, pe_names)

        sim_iteration = iterations[0]
        end_iteration = iterations[1]
        timeline = self.timeline
        occupy_times = self.occupy_times
        sched = Schedule(self.num_pe)
        # Start scheduling simulation
        while sim_iteration < end_iteration:
            t = timeline.pop(0)
            self._enqueue(t)

            # Check every PE's ready_queue(Priority queue)
            for pe_idx, q in enumerate(self._ready_queues):
                if occupy_times[pe_idx] > t or q.size() == 0:
                    continue

                l, layer_idx, pe, app = self._pop_and_get_layer_info(q)

                execution_time, transition_time, transition_time_list = app.do_layer(l, pe, t)
                end_time = t + execution_time
                occupy_times[pe_idx] = end_time + transition_time

                # Update iteration's end time
                self._set_pe_time(pe, l.iteration, t, occupy_times[pe_idx])

                self.iteration[layer_idx] = self.iteration[layer_idx] + 1

                # self._update_timeline(l, occupy_times[pe_idx])
                if transition_time_list == []:
                    self._update_timeline(l, occupy_times[pe_idx])
                else:
                    for time in transition_time_list:
                        self._update_timeline(l, time)

                l.increase_iter()

                if l.iteration <= 1:
                    time_tuple = (pe, l, t, end_time, transition_time)
                    sched.add_sched(time_tuple)

                # FIXME What is second condition?
                if draw_gantt and occupy_times[pe_idx] != t and l.iteration <= self.draw_iteration:
                    time_tuple = (l.get_name(), self.pe_list[pe].name, t, end_time, transition_time)
                    gantt.add_task(time_tuple)

                self.elapsed_time_per_pe[pe_idx] += (end_time - t)

                if l.is_end_node and l.iteration == 1:
                    self.response_time[self.app_list.index(app)] = end_time

            inc_sim_iteration = True
            for n in self.iteration:
                if n < end_iteration:
                    inc_sim_iteration = False
            if inc_sim_iteration:
                sim_iteration += 1

        if draw_gantt:
            self._draw_gantt(gantt, gantt_name, mapping, fitness)

        return sched

    def get_response_time(self, app):
        return self.response_time[self.app_list.index(app)]

