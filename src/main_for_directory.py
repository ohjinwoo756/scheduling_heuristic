

import sys
import os
import google.protobuf.text_format
import caffe_pb2
from layer_graph import LayerGraph
from Layer import Layer, LayerType
from app import Application
from util import *
from sched_simulator import SchedSimulator
from pe import PE
import time

import config  # Utilization variable shared across modules.
from plot import Plot # For making plot


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
    config.name = ""
    for n in config.nets:
        config.name += n
        config.name += "_"
    config.net_files = []
    config.est_files = []
    config.sched_method = args.sched_method

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
    config.processor = args.processor
    # if args.hyper_parameter != None:
    #     config.hyper_parameter = float(args.hyper_parameter)
    # else:
    #     config.hyper_parameter = None


def parse_result_path(args):
    path_name_processor = ""
    path_name_nets = ""
    path_name_cpu_config = str(config.cpu_config)
    path_name_period = ""

    for idx, p in enumerate(config.processor):
        path_name_processor += "{}_".format(str(p))
    for idx, n in enumerate(config.nets):
        path_name_nets += "{}_".format(str(n))
        path_name_period += "{}_".format(str(config.period[idx]))

    path_name_processor = path_name_processor[:-1]
    path_name_nets = path_name_nets[:-1]
    path_name_period = path_name_period[:-1]

    config.save_path = args.save_path
    if not os.path.exists(config.save_path + path_name_processor):
        os.mkdir(config.save_path + path_name_processor)
    config.save_path += path_name_processor + "/"
    if not os.path.exists(config.save_path + path_name_nets):
        os.mkdir(config.save_path + path_name_nets)
    config.save_path += path_name_nets + "/"
    if not os.path.exists(config.save_path + path_name_cpu_config):
        os.mkdir(config.save_path + path_name_cpu_config)
    config.save_path += path_name_cpu_config + "/"
    if not os.path.exists(config.save_path + path_name_period):
        os.mkdir(config.save_path + path_name_period)
    config.save_path += path_name_period


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
    parser.add_argument("-s", "--schedule_method", choices=config.schedulers, dest="sched_method", help="Scheduler to use. Valid scheduler choices are {}".format(config.schedulers))
    parser.add_argument("-n", "--network", nargs='+', choices=config.networks, dest="net", help="Network prototxt file to use.")
    parser.add_argument("-c", "--cpu_core_distribution", choices=config.CPU_intraParall, dest="cpu_core_distribution", help="PE configuration to apply. Valid choices are {}".format(config.CPU_intraParall))
    # parser.add_argument("-r", "--priority", nargs='+', dest="priority", help="Priority to assign. Number 1 has the best priority.")
    parser.add_argument("-o", "--objective", choices=config.app_to_obj_dict, nargs='+', dest="objective", help="Objective to apply. Valid choices are {}".format(config.app_to_obj_dict))
    parser.add_argument("-t", "--constraint", choices=config.app_to_cst_dict, nargs='+', dest="constraint", help="Constraint to apply. Valid choices are {}".format(config.app_to_cst_dict))
    parser.add_argument("-u", "--cpu_utilization", choices=config.CPU_util, dest="cpu_utilization", default=100, help="CPU utilization to apply. Valide choices are {}".format(config.CPU_util))
    parser.add_argument("--opt_energy", dest="opt_energy", default=False, action="store_true", help="Optimize energy.")
    parser.add_argument("-e", "--energy_cst", dest="energy_cst", default=0, help="Set energy constraint")
    parser.add_argument("-d", "--period", nargs='*', dest="period", help="Period to apply. The unit is micro-second(us)")
    parser.add_argument("-p", "--save_path", dest="save_path", default="./", help="set destination folder path to save result files", metavar="DIR")
    parser.add_argument("-i", "--est_type", choices=config.est_type, dest="est_type", default="basic", help="set type of estimation input")
    # TODO need to restrict rt and mobility
    parser.add_argument("-a", "--analyzer", dest="analyzer", default="mobility", help="set analyzer")
    parser.add_argument("-r", "--processor", nargs='+', default=['cpu', 'gpu', 'npu'], dest="processor", help="PE set to be scheduled.")
    # XXX: only for JHeuristic
    # parser.add_argument("-y", "--hyper_parameter", default=None, dest="hyper_parameter", help="Set hyper parameter in JHeuristic.")

    args = parser.parse_args()
    check_requirement(args, parser, required)
    copy_arg2config(args)

    return args


if __name__ == '__main__':
    options = parse_options()
    parse_result_path(options)


