from collections import defaultdict
from orderedset import OrderedSet
from Edge import Edge
from Layer import LayerType
from merged_layer import MergedLayer


class DirectedGraph(object):
    """ Directed Graph data structure """

    def __init__(self, connections):
        self._graph = defaultdict(set)
        self.add_connections(connections)

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """
        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """
        self._graph[node1].add(node2)

    def remove(self, node):
        """ Remove all references to node """
        for _, cxns in self._graph.iteritems():
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """
        return node1 in self._graph and node2 in self._graph[node1]

    def find_path(self, node1, node2, path=[]):
        """ Find any path between node1 and node2 (may not be shortest) """
        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))

    def num_node(self):
        return len(self._graph)


class LayerGraph(DirectedGraph):
    def __init__(self, connections):
        self._in_edge = defaultdict(set)
        self._out_edge = defaultdict(set)
        self._edge = defaultdict(Edge)
        self._buffer_size = 1
        super(LayerGraph, self).__init__(connections)

    def add(self, node1, node2):
        self._graph[node1].add(node2)
        e = Edge(self._buffer_size, node1, node2)
        self._in_edge[node2].add(e)
        self._out_edge[node1].add(e)

        self._edge[(node1, node2)] = e

    def do_init(self, layer_list):
        for e in self._edge.values():
            e.do_init()

        for n in layer_list:
            n.do_init()

    def set_edge_type(self):
        from pe import PEType
        for src, dst_set in self._graph.iteritems():
            src_pe = src.pe
            for dst in dst_set:
                dst_pe = dst.pe

                if src_pe == dst_pe:
                    continue

                # If memory transition exists,
                self._edge[(src, dst)].transition_flag = True
                if src_pe.get_type() == PEType.CPU and dst_pe.get_type() == PEType.GPU:
                    self._edge[(src, dst)].cpu2gpu = True
                elif src_pe.get_type() == PEType.GPU and dst_pe.get_type() == PEType.CPU:
                    self._edge[(src, dst)].gpu2cpu = True
                elif src_pe.get_type() == PEType.CPU and dst_pe.get_type() == PEType.NPU:
                    self._edge[(src, dst)].cpu2npu = True
                elif src_pe.get_type() == PEType.NPU and dst_pe.get_type() == PEType.CPU:
                    self._edge[(src, dst)].npu2cpu = True
                elif set([src_pe.get_type(), dst_pe.get_type()]) == set([PEType.GPU, PEType.NPU]):
                    self._edge[(src, dst)].gpu_npu_connection = True

    # for priority
    def set_dfs_priority(self, node):
        prio_array = self._set_dfs_priority(node)
        # print(prio_array) # '_' has everything in dfs order
        for l in prio_array:
            l.set_increase_prio()
            # print(l.priority) # 0 ~ .. => dfs order first

    def _set_dfs_priority(self, node, visited_node=set([]), stack=[]):
        visited_node.add(node)

        for n in self._graph[node]:
            if n not in visited_node:
                _ = self._set_dfs_priority(n, visited_node)

        stack.insert(0, node)
        return stack

    def propagate_mobility(self, node, mobility):
        idx = 0
        visited_node=set([node])
        update_nodes = [n for n in self._graph[node] if n not in visited_node and n.mobility > mobility]

        while len(update_nodes) > idx:
            _node = update_nodes[idx]
            if _node not in visited_node:
                temp = [n for n in self._graph[_node] if n not in visited_node and n.mobility > mobility]
                _node.set_mobility(mobility)
                update_nodes.extend(temp)
                visited_node.add(_node)
            idx += 1

    def _check_all_in_edge_pe(self, node):
        flag = True
        for in_edge in self._in_edge[node]:
            if in_edge.sender.get_pe() != node.get_pe():
                flag = False
                break
        return flag

    def _get_pe_cluster(self, node):
        idx = 0
        add_node = [n for n in self._graph[node] if self._check_all_in_edge_pe(n)]

        while len(add_node) > idx:
            _node = add_node[idx]
            temp = [n for n in self._graph[_node] if self._check_all_in_edge_pe(n)]
            add_node.extend(temp)
            idx += 1

        return add_node

    def _merge_node(self, node_list):
        _node_set = set([])
        merged_list = []

        for n in node_list:
            if n not in _node_set:
                node_set = OrderedSet([n])
                node_set.update(self._get_pe_cluster(n))
                _node_set.update(node_set)
                if merged_list and node_set.intersection(merged_list[-1]):
                    merged_list[-1].update(node_set)
                else:
                    merged_list.append(node_set)

        return merged_list

    def _merged_connections(self, nodes_list):
        merge_node_list = []
        layer2merged = {}
        for nodes in nodes_list:
            m = MergedLayer(nodes)
            merge_node_list.append(m)
            for l in nodes:
                layer2merged[l] = m

        connections = set([])
        for idx1, nodes1 in enumerate(nodes_list):
            for n in nodes1:
                m1 = layer2merged[n]
                for child in self._graph[n]:
                    m2 = layer2merged[child]
                    if m2 is not m1:
                        connections.add((m1, m2))
        return merge_node_list, connections

    def simplify_graph(self, node):
        _paths = self.get_all_path(node, [[]])
        _paths.sort(key=len, reverse=True)
        del _paths[-1]  # remove empty path

        remove_idx = []
        for idx1, path1 in enumerate(_paths):
            if idx1 in remove_idx:
                continue

            for idx2 in range(idx1 + 1, len(_paths)):
                if set(_paths[idx2]).issubset(set(path1)):
                    remove_idx.append(idx2)

        for idx in sorted(remove_idx, reverse=True):
            del _paths[idx]

        connections = set([])
        for p in _paths:
            prev = None
            for l in p:
                if prev is None:
                    prev = l
                    continue
                connections.add((prev, l))
                prev = l

        return LayerGraph(connections)

    def merge_graph(self, node_list):
        merged_list = self._merge_node(node_list)
        merge_list, connections = self._merged_connections(merged_list)
        return merge_list[0], LayerGraph(connections)

    def get_all_path(self, node, paths=[[]]):
        paths[-1].append(node)
        if 0 == len(self._graph[node]):
            paths.append(list(paths[-1]))
            paths[-1].pop()
            return paths

        for n in self._graph[node]:
            paths = self.get_all_path(n, paths)

        paths[-1].pop()
        return paths

    # for simulation
    def check_runnable(self, node, t):
        for in_edge in self._in_edge[node]:
            schedulable, _in_check_sched_flag = in_edge.check_in_buffer(t)
            if not schedulable:
                node.need_in_edge_check = _in_check_sched_flag
                return False

        for out_edge in self._out_edge[node]:
            if not out_edge.check_out_buffer():
                node.need_out_edge_check = False
                return False
        return True

    def get_transition_time(self, node):
        transition_time = 0
        for out_edge in self._out_edge[node]:
            transition_time += out_edge.calc_transition_time()
        return transition_time

    def do_layer(self, node, pe, t):
        for in_edge in self._in_edge[node]:
            in_edge.consume_data()

        exec_time = 0
        if node.time_list is not None:
            exec_time = node.time_list[pe]

        transition_time = 0
        for out_edge in self._out_edge[node]:
            transition_time += out_edge.calc_transition_time()
            out_edge.produce_data(t + exec_time + transition_time)

        return exec_time, transition_time
