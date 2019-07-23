##
# @file main.py
# @brief main entrance
# @details Input parsing and application starting point.

import sys
import google.protobuf.text_format
import caffe_pb2
from layer_graph import LayerGraph
from Layer import Layer, LayerType
from app import Application
from util import *
from sched_simulator import SchedSimulator
from pe import PE

import config  # Utilization variable shared across modules.


def automatic_search_for_networks_and_inputs(input_dir_path):
    import os
    try:
        filenames = os.listdir(input_dir_path)
        for filename in filenames:
            full_filename = os.path.join(input_dir_path, filename)
            if os.path.isdir(full_filename):
                config.networks.append(filename)  # available network names
                automatic_search_for_networks_and_inputs(full_filename)  # recursive
            else:
                splited_string = filename.split('_')
                for s in splited_string:
                    if s == "cfg":
                        # full cfg file name
                        config.cfg_prototxt_path.append(full_filename)
                        break
                    elif s == "estimation":
                        # full estimation file name
                        config.est_prototxt_path.append(full_filename)
                        # print config.est_prototxt_path
                        break
                    else:
                        pass
    except PermissionError:
        pass


def copy_arg2config(args):
    config.app_to_obj_dict = [obj for obj in args.objective]
    config.app_to_cst_dict = [cst for cst in args.constraint]
    config.CPU_UTILIZATION = int(args.cpu_utilization)
    config.opt_energy = args.opt_energy
    config.energy_cst = int(args.energy_cst)
    config.nets = args.net
    config.net_files = []
    config.est_files = []

    config.cpu_config = args.cpu_core_distribution
    config.num_virtual_cpu = len(config.cpu_config)

    total_cpu_core_num = 0
    for cpu_pe in range(0, len(config.cpu_config)):
        core_num = int(config.cpu_config[cpu_pe])
        total_cpu_core_num += core_num
    config.num_pe = total_cpu_core_num

    config.period = [int(p) for p in args.period]
    config.priority = args.priority
    for n in config.nets:
        for f in config.cfg_prototxt_path:
            if n in f:
                config.net_files.append(f)
                break
        for f in config.est_prototxt_path:
            if args.est_type == "basic":
                if n in f and "_%s.prototxt" % args.cpu_core_distribution in f:
                    config.est_files.append(f)
            else:
                if n in f and "_%s_%s.prototxt" % (args.cpu_core_distribution, \
                        args.est_type) in f:
                    config.est_files.append(f)

    config.analyzer = args.analyzer


def check_requirement(args, parser, required):
    # check requirement
    for r in required:
        if args.__dict__[r] is None:
            parser.error("parameter '%s' required" % r)
        elif r == "net":
            config.num_of_app = len(args.net)

    if len(args.objective) != config.num_of_app:
        parser.error("Objective array need (# of application) items")

    if args.__dict__["constraint"] is None:
        args.constraint = ["None"] * config.num_of_app
    elif len(args.constraint) != config.num_of_app:
        parser.error("Constraint array need (# of application) items")

    if args.__dict__["period"] is None:
        args.period = [0 for _ in enumerate(args.net)]
    elif len(args.period) != config.num_of_app:
        parser.error("Period array need (# of application) items")

    args.priority = [i + 1 for i, _ in enumerate(args.net)]
    # if args.__dict__["priority"] is None:
    #     args.priority = [i + 1 for i, _ in enumerate(args.net)]
    # elif len(args.priority) != config.num_of_app:
    #     parser.error("Priority array need (# of application) items")


def parse_options():
    # from optparse import OptionParser
    from argparse import ArgumentParser

    # Search model & estimation prototxt
    automatic_search_for_networks_and_inputs(config.input_dir_path)

    usage = """prog [options]"""
    parser = ArgumentParser(usage=usage)

    required = "net cpu_core_distribution objective".split()

    # TODO add energy object and constraint
    parser.add_argument("-s", "--schedule_method", choices=config.schedulers, dest="sched_method", default="GA", help="Scheduler to use. Valid scheduler choices are {}".format(config.schedulers))
    parser.add_argument("-n", "--network", nargs='+', choices=config.networks, dest="net", help="Network prototxt file to use.")
    parser.add_argument("-c", "--cpu_core_distribution", choices=config.CPU_intraParall, dest="cpu_core_distribution", help="PE configuration to apply. Valid choices are {}".format(config.CPU_intraParall))
    # parser.add_argument("-r", "--priority", nargs='+', dest="priority", help="Priority to assign. Number 1 has the best priority.")
    parser.add_argument("-o", "--objective", choices=config.app_to_obj_dict, nargs='+', dest="objective", help="Objective to apply. Valid choices are {}".format(config.app_to_obj_dict))
    parser.add_argument("-t", "--constraint", choices=config.app_to_cst_dict, nargs='+', dest="constraint", help="Constraint to apply. Valid choices are {}".format(config.app_to_cst_dict))
    parser.add_argument("-u", "--cpu_utilization", choices=config.CPU_util, dest="cpu_utilization", help="CPU utilization to apply. Valide choices are {}".format(config.CPU_util), default=100)
    parser.add_argument("--opt_energy", dest="opt_energy", default=False, action="store_true", help="Optimize energy.")
    parser.add_argument("-e", "--energy_cst", dest="energy_cst", default=0, help="Set energy constraint")
    parser.add_argument("-d", "--period", nargs='*', dest="period", help="Period to apply. The unit is micro-second(us)")
    parser.add_argument("-p", "--save_path", dest="save_path", default="./", help="set destination folder path to save result files", metavar="DIR")
    parser.add_argument("-i", "--est_type", choices=config.est_type, dest="est_type", default="basic", help="set type of estimation input")
    # TODO need to restrict rt and mobility
    parser.add_argument("-a", "--analyzer", dest="analyzer", default="rt", help="set analyzer")

    args = parser.parse_args()

    check_requirement(args, parser, required)

    copy_arg2config(args)
    return args


def read_network_prototxt(network_prototxt):
    net = caffe_pb2.NetParameter()
    f = open(network_prototxt, "r")
    google.protobuf.text_format.Merge(f.read(), net)
    f.close()
    return net


def read_estimation(estimation_prototxt):
    est = caffe_pb2.NetParameter()
    f = open(estimation_prototxt, "r")
    google.protobuf.text_format.Merge(f.read(), est)
    f.close()
    return est


def make_layer(idx, name, l, name2layer):
    bottom_layers = []
    bottom_layer = None
    if l.bottom:
        for bottom_name in l.bottom:
            bottom_layers.append(name2layer["app_{} {}".format(idx, bottom_name)])
        if len(bottom_layers) == 1:
            bottom_layer = bottom_layers[0]

    if l.type == "Input":
        layer = Layer(name, LayerType.FRONT, l.input_param.shape[0].dim[1], l.input_param.shape[0].dim[2:], is_start_node=True)
    elif l.type == "Convolution":
        conv_param = l.convolution_param
        size = bottom_layer.get_size()
        kernel_size = int(conv_param.kernel_size[0])
        if conv_param.pad:
            pad = int(conv_param.pad[0])
        else:
            pad = 0
        if conv_param.stride:
            stride = int(conv_param.stride[0])
        else:
            stride = 1
        layer = Layer(name, LayerType.CONV, conv_param.num_output, size, kernel_size, pad, stride)
    elif l.type == "Pooling":
        pool_param = l.pooling_param
        size = bottom_layer.get_size()
        kernel_size = int(pool_param.kernel_size)
        if pool_param.pad:
            pad = int(pool_param.pad[0])
        else:
            pad = 0
        if pool_param.stride:
            stride = int(pool_param.stride)
        else:
            stride = 1
        layer = Layer(name, LayerType.POOL, bottom_layer.get_num_output(), size, kernel_size, pad, stride)
    elif l.type == "InnerProduct":
        layer = Layer(name, LayerType.FULLY, [l.inner_product_param.num_output, 1, 1])
    elif l.type == "Shortcut":
        layer = Layer(name, LayerType.SHORTCUT, l.shortcut_param.num_output[0], bottom_layers[0].get_size())
    elif l.type == "Concat":
        num_output = 0
        for l in bottom_layers:
            num_output += l.num_output
        layer = Layer(name, LayerType.CONCAT, num_output, bottom_layers[0].get_size())
    elif l.type == "Softmax":
        layer = Layer(name, LayerType.BACK, bottom_layer.get_num_output(), bottom_layer.get_size(), is_end_node=True)
    else:
        raise TypeError("Not supported type : " + l.type)

    return layer


def make_layer_structure(idx, net):
    connections = []
    name2layer = {}
    layer_list = []

    for l in net.layer:
        name = "app_{} {}".format(idx, l.name)
        layer = make_layer(idx, name, l, name2layer)
        name2layer[name] = layer
        layer_list.append(layer)
        for b in l.bottom:
            connections.append((name2layer["app_{} {}".format(idx, b)], layer))

    for l in layer_list:
        if l.layer_type != LayerType.CONCAT:
            continue

        reverse_conn = reversed(connections)
        in_layers = [c[0] for c in reverse_conn if c[1] is l]
        out_layers = [c[1] for c in connections if c[0] is l]
        for in_l in in_layers:
            connections.remove((in_l, l))  # remove connection between input - reshape(concat)
            for out_l in out_layers:
                connections.append((in_l, out_l))  # add new connection between input and output
        for out_l in out_layers:
            connections.remove((l, out_l))  # remove connection between reshape(concat) - output

    layer_graph = LayerGraph(connections) # XXX: layer precedences applied to 'layer_graph'
    return layer_graph, layer_list, name2layer


def make_estimation(idx, est, name2layer, pe_list):
    for l in est.layer:
        name = l.name
        layer = name2layer["app_{} {}".format(idx, name)]
        layer_time = []
        for v in pe_list:
            # FIXME there is assumption
            if 'cpu1' in v.name:
                # In profile input, the first cpu is represented by 'cpu', not 'cpu1'
                # getattr(l, 'cpu') extract 'cpu' attribute from estimation prototxt
                layer_time.append(getattr(l, 'cpu'))
            elif 'cpu' in v.name:
                layer_time.append(getattr(l, v.name[:4]))
            else: # gpu, npu, ...
                layer_time.append(getattr(l, v.name[:3]))

        layer.set_time_list(layer_time)


# XXX deprecated
def get_processor_array():
    processor_array = []
    for i, pe_name in enumerate(config.processor):
        processor_array.append(pe_name)
        if pe_name == 'cpu':
            for i in range(1, config.num_virtual_cpu):
                processor_array.append(pe_name + str(i + 1))
    return processor_array


def init_processors():
    pe_list = []
    # XXX
    # there must be more than one cpu proceessor.
    for i in range(config.num_virtual_cpu):
        pe_list.append(PE('cpu' + str(i + 1) + ' (' + str(config.cpu_config[i]) + ')'))

    # FIXME there is assumption
    # pe type equal to pe name except cpu
    for i, pe_type in enumerate(config.processor):
        if 'cpu' in pe_type:
            continue
        pe_list.append(PE(pe_type))

    return pe_list


if __name__ == '__main__':
    options = parse_options()

    # Load scheduler
    mapping_module = __import__(options.sched_method) # Dynamic import != Static import
    mapping_class = getattr(mapping_module, options.sched_method) # (module name, class name)

    # Read network and estimation file
    nets = []
    ests = []
    for idx, f in enumerate(config.net_files):
        nets.append(read_network_prototxt(f))
        ests.append(read_estimation(config.est_files[idx]))

    # processor_array = get_processor_array()
    pe_list = init_processors()

    # Make structure
    layer_lists = []
    app_list = []
    prev_concat_num = 0
    for idx, n in enumerate(nets):
        layer_graph, layer_list, name2layer = make_layer_structure(idx, n) # layer precedences applied. dict: layer_graph._graph[node1] => node2
        make_estimation(idx, ests[idx], name2layer, pe_list) # apply profile inputs -> layer.time_list
        layer_graph.set_dfs_priority(layer_list[0]) # XXX later ...
        layer_lists.append(layer_list) # set of layers 
        app = Application(config.nets[idx], layer_graph, layer_list, config.period[idx], config.priority[idx], prev_concat_num)
        prev_concat_num += app.get_num_concat()
        app_list.append(app) # set of apps

    layer_list = [j for i in layer_lists for j in i]

    # Initialize scheduler class
    mapper = mapping_class(app_list, pe_list)
    # XXX deprecated
    from ILP import ILP
    if isinstance(mapper, ILP):
        mapper.set_options(options)

    # scheduling
    mappings = mapper.do_schedule()

    # for Gantt chart (not fitness calculation)
    sched_sim = SchedSimulator(app_list, pe_list)
    # draw gantt
    for m in mappings:
        PE.init_apps_pe_by_mapping(app_list, m, pe_list)
        sched_sim.do_init()
        name = ""
        for n in config.nets:
            name += n
            name += "_"
        objs = []
        csts = []
        for obj in mapper.fitness.objs:
            objs.append(type(obj).__name__)
        for cst in mapper.fitness.csts:
            csts.append(type(cst).__name__)
        gantt_name = "{}_{}_{}_{}_{}_{}_{}.png".format(options.save_path + "/" + name, str(config.priority), str(config.period), str(options.cpu_core_distribution), str(objs), str(csts), config.analyzer)
        sched_sim.do_simulation(m, (0, 5), True, gantt_name, mapper.fitness)
