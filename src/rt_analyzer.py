from analyzer import Analyzer
import itertools
# from demand_function import DemandFunction


class Task(object):
    prio = itertools.count().next

    def __init__(self, layer):
        self.layer = layer
        self._app = layer.get_app()
        self.do_init()

    def do_init(self):
        self._offset = 0
        self._deadline = -1.
        self._priority = -1
        self._exec_time = -1.

    def get_app(self):
        return self._app

    def get_pe(self):
        return self._pe

    def set_pe(self, pe):
        self._pe = pe
        self._exec_time = self.layer.time_list[pe]

    def get_exec(self):
        return self._exec_time

    def get_offset(self):
        return self._offset

    def get_deadline(self):
        return self._deadline

    @staticmethod
    def sort(a, b):  # deadline -> exec time -> chromosome index
        if a._deadline != b._deadline:
            return -1 if a._deadline - b._deadline < 0 else 1  # increase order
        elif a._exec_time != b._exec_time:
            return -1 if b._exec_time - a._exec_time < 0 else 1  # decrease order
        else:
            return a.layer.get_index() - b.layer.get_index()  # increase order

    def get_priority(self):
        return self._priority

    def get_period(self):
        return self.layer.get_period()

    def set_priority(self):
        self._priority = Task.prio()

    def set_time(self, offset, deadline):
        self._offset = offset
        # self.layer.set_offset(offset)
        self._deadline = max(deadline, 0)
        # print(self.layer.name, offset, self._deadline)
        # print("====================================")

    def __repr__(self):
        return repr('name: {}({}) | exec: {} | offset: {} | deadline: {}'.format(self.layer.name, self._priority, self._exec_time, self._offset, self._deadline))

    def __str__(self):
        return 'name: {}({})\t| exec: {}\t| offset: {}\t| deadline: {}'.format(self.layer.name, self._priority, self._exec_time, self._offset, self._deadline)


class Path(object):
    _task_dict = {}

    def __init__(self, _task_list, deadline=-1):
        self.exec_time = 0
        if deadline == -1:
            task_list = []
            for _task in _task_list:
                task = None
                if _task in Path._task_dict:
                    task = Path._task_dict[_task]
                else:
                    task = Task(_task)
                    Path._task_dict[_task] = task
                task_list.append(task)
            self._deadline = _task_list[0].get_period()
        else:
            self._deadline = deadline
            task_list = _task_list
        self._task_list = task_list

    def contain_task(self, task):
        if task in self._task_list:
            return True
        else:
            return False

    def get_deadline(self):
        return self._deadline

    def get_task_list(self):
        return self._task_list

    def set_exec(self, time):
        self.exec_time = time

    def calc_exec(self):
        for _task in self._task_list:
            self.exec_time += _task.get_exec()

    def _have_all_task_deadline(self):
        for task in self._task_list:
            if task.get_deadline() != -1:
                return False
        return True

    def set_time(self, path_offset=0):
        task_list = self._task_list
        offset = path_offset

        if self._have_all_task_deadline():
            path_deadline = self._deadline
            slack = max((path_deadline - self.exec_time), 0) / len(task_list)  # PURE
            for task in task_list:
                if task is task_list[-1]:
                    task.set_time(offset, (path_deadline + path_offset) - offset)
                else:
                    task.set_time(offset, task.get_exec() + slack)
                    offset += task.get_exec() + slack
        else:
            _task_list = []
            for task in task_list:
                if task.get_deadline() == -1:
                    _task_list.append(task)
                elif _task_list:  # task_list is not empty
                    p = Path(_task_list, task.get_offset() - offset)
                    p.calc_exec()
                    p.set_time(offset)
                    offset = task.get_offset() + task.get_deadline()
                    del _task_list[:], p
                else:
                    offset += task.get_deadline()


class RTAnalyzer(Analyzer):
    instance = {}

    def __new__(cls, *args, **kwargs):
        app = args[1]
        if app in RTAnalyzer.instance:
            return RTAnalyzer.instance[app]
        inst = super(RTAnalyzer, cls).__new__(cls, *args, **kwargs)
        RTAnalyzer.instance[app] = inst
        inst.is_init = False
        return inst

    def __init__(self, app_list, app, pe_list):
        if self.is_init:
            return

        self.app_list = app_list
        self.app = app
        self.pe_list = pe_list
        self.path_list = {}
        self.task_list = []
        self.is_init = True

    def _construct_path(self):
        self.task_list = []
        for app in self.app_list:
            app.do_init()
            g = app.simplified_graph
            merge_first_node, g = g.merge_graph(app.layer_list)
            # _paths = g.get_all_path(app.layer_list[0], [[]])
            _paths = g.get_all_path(merge_first_node, [[]])
            del _paths[-1]  # remove empty path
            _paths.sort(key=len, reverse=True)

            path_list = []
            task_set = set()
            for path in _paths:
                path_list.append(Path(path))
                task_set.update(path_list[-1].get_task_list())

            task_list = list(task_set)
            self.path_list[app] = path_list
            self.task_list.extend(task_list)
            del _paths, task_set, task_list, g

    def _set_deadline(self):
        for app in self.app_list:
            for path in self.path_list[app]:
                path.set_time()

    def _map_task2pe(self, mapping):
        for task in self.task_list:
            idx = task.layer.get_index()
            task.set_pe(mapping[idx])

    def _set_path_exec(self):
        for app in self.app_list:
            for path in self.path_list[app]:
                path.calc_exec()
            self.path_list[app] = sorted(self.path_list[app], key=lambda path: path.exec_time, reverse=True)
            # print("{} Critical Path {} Path Lenth {}".format(app.name, self.path_list[app][0].exec_time, len(self.path_list[app][0]._task_list)))

    def _set_priority(self):
        # self.task_list = sorted(self.task_list, cmp=Task.sort)
        for app in self.app_list:
            for path in self.path_list[app]:
                for task in path._task_list:
                    if task.get_priority() == -1:
                        task.set_priority()

    def _init_all(self):
        for app in self.app_list:
            for path in self.path_list[app]:
                path.set_exec(0)

        for task in self.task_list:
            task.do_init()

        Task.prio = itertools.count().next

    def _merge_tasks(self):
        pass

    def preprocess(self, mapping):
        self._construct_path()
        self._init_all()
        self._map_task2pe(mapping)
        self._set_path_exec()
        self._set_deadline()
        self._set_priority()
        self.app.graph.set_edge_type()
        # for a in self.app_list:
        #     if a.layer_list[1].offset != -1:
        #         DemandFunction(a, mapping, self.pe_list)
        # print("===================================")
        # for task in self.task_list:
        #     print(task)
        # print("===================================")

    def _contain_tasks(self, task1, task2):
        app = task1.get_app()
        if task1.get_app() is not task2.get_app():
            return False

        for path in self.path_list[app]:
            if path.contain_task(task1) and path.contain_task(task2):
                return True
        return False

    def _get_interference(self, target):
        hp_list = []
        lp_list = []
        # print(target.layer.name + " " + str(target.get_priority()))
        for task in self.task_list:
            if task.get_pe() != target.get_pe() or self._contain_tasks(task, target):
                continue
            # print("\t" + task.layer.name + " " + str(task.get_priority()))
            if task.get_priority() < target.get_priority():
                hp_list.append(task)
            elif task.get_priority() > target.get_priority():
                lp_list.append(task)

        # print(lp_list)
        lp_interference = reduce((lambda x, y: y.get_exec() if x < y.get_exec() else x), lp_list, 0)

        import math
        hp_interference = reduce((lambda x, y: math.ceil(target.get_deadline() / y.get_period()) * y.get_exec() + x), hp_list, 0)

        return lp_interference, hp_interference

    def get_response_time(self):
        app = self.app
        worst_response = 0.
        for target in self.task_list:
            l = target.layer
            if l.get_app() is not app:
                continue

            lp_interference, hp_interference = self._get_interference(target)
            transfer_time = l.transfer_time
            worst_exec = target.get_exec() + lp_interference + hp_interference
            worst_response += worst_exec + transfer_time
            # print("Name {app.name:>12s}\tPeriod {app.period:.2f}\tWorst Exec {worst_exec:.2f}\tTransfer Time {transfer_time:.2f}\tInterference ({lp_interference:.2f}, {hp_interference:.2f})"
            #       .format(app=app, worst_exec=worst_exec, transfer_time=transfer_time, lp_interference=lp_interference, hp_interference=hp_interference))

        Path._task_dict.clear()                
        for task in self.task_list:            
            del task.layer                     
        del self.task_list[:]                  
        for a, p in self.path_list.iteritems():
                del p                              
        self.path_list.clear()                 

        return worst_response

    def get_schedulable_penalty(self):
        app = self.app
        penalty = 0
        for target in self.task_list:
            l = target.layer
            if l.get_app() is not app:
                continue

            lp_interference, hp_interference = self._get_interference(target)
            worst_exec = target.get_exec() + lp_interference + hp_interference
            # transfer_time = app.graph.get_transition_time(l, l.get_pe())
            penalty += max(worst_exec - target.get_deadline(), 0)
            # print(target.layer.name, target.get_deadline(), worst_exec, target.get_exec(), lp_interference, hp_interference)

        Path._task_dict.clear()                
        for task in self.task_list:            
            del task.layer                     
        del self.task_list[:]                  
        for a, p in self.path_list.iteritems():
                del p                              
        self.path_list.clear()                 
        return 1024 * penalty

    def get_violated_layer(self):
        raise NotImplementedError
        app = self.app
        violated_layer = None
        for target in self.task_list:
            l = target.layer
            if l.get_app() is not app:
                continue

            lp_interference, hp_interference = self._get_interference(target)
            worst_exec = target.get_exec() + lp_interference + hp_interference
            # transfer_time = app.graph.get_transition_time(l, l.get_pe())
            if worst_exec - target.get_deadline() > 0:
                violated_layer = target.layer
                break

        Path._task_dict.clear()                
        for task in self.task_list:            
            del task.layer                     
        del self.task_list[:]                  
        for a, p in self.path_list.iteritems():
                del p                              
        self.path_list.clear()                 

        return violated_layer
