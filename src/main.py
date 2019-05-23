import cmd_line_option_parser
import input_reader

# This application deals with only 1 dag yet.
if __name__ == '__main__':

    # Parse command line option
    options = cmd_line_option_parser.parse_options()
    # Read network inputs to make caffe_pb2.NetParameter variable
    networks = []
    profiles = []
    for idx in range(0, len(options.networks)):
        networks.append(input_reader.read_network_prototxt(options.networks[idx]))
        profiles.append(input_reader.read_network_profile_prototxt(options.profiles[idx]))
    # debug 
    # print networks[0].layer[6].bottom
    # print networks[0].layer[6].top
    # print networks[1].layer[0].name
    # print profiles[1].layer[0].cpu

    # Load scheduler
    scheduler_module = __import__(options.scheduler) # Dynamic import != Static import
    scheduler_class = getattr(scheduler_module, options.scheduler) # (module name, class name)
    # The right below also possible
    # scheduler_class = getattr(scheduler_module, options.scheduler)(networks, profiles)
    scheduler_object = scheduler_class(networks, profiles) # Make class object

    # Scheduling
    scheduler_object.do_schedule()

    # Visualization (Gantt chart)

    
