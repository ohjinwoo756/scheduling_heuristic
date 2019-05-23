import cmd_line_option_parser
import input_reader

def read_in_NetParameter(options):
    networks = []
    profiles = []
    for idx in range(0, len(options.networks)):
        networks.append(input_reader.read_network_prototxt(options.networks[idx]))
        profiles.append(input_reader.read_network_profile_prototxt(options.profiles[idx]))
    return networks, profiles

def make_scheduler_object(options, networks, profiles):
    # Load scheduler
    scheduler_module = __import__(options.scheduler) # Dynamic import != Static import
    scheduler_class = getattr(scheduler_module, options.scheduler) # (module name, class name)
    # The right below also possible
    # scheduler_class = getattr(scheduler_module, options.scheduler)(networks, profiles)
    scheduler_object = scheduler_class(networks, profiles) # Make class object

    return scheduler_object

if __name__ == '__main__':

    # Parse command line option
    options = cmd_line_option_parser.parse_options()

    # Read inputs in caffe_pb2.NetParameter format
    networks, profiles = read_in_NetParameter(options)
    # debug 
    # print networks[0].layer[6].bottom
    # print networks[0].layer[6].top
    # print networks[1].layer[0].name
    # print profiles[1].layer[0].cpu

    scheduler_object = make_scheduler_object(options, networks, profiles)

    # Scheduling
    scheduler_object.do_schedule()

    # Visualization (Gantt chart)

    
