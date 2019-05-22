import cmd_line_option_parser
import input_reader

# This application deals with only 1 dag yet.
if __name__ == '__main__':

    # Parse command line option
    options = cmd_line_option_parser.parse_options()

    # Read network information to make caffe_pb2.NetParameter variable
    network = input_reader.read_network_prototxt(options.network)
    profile = input_reader.read_network_profile_prototxt(options.profile)
    # print network.layer[0].name
    # print profile.layer[0].cpu

    # Load scheduler
    scheduler_module = __import__(options.scheduler) # Dynamic import
    scheduler_class = getattr(scheduler_module, options.scheduler) # (module name, class name)
    scheduler_object = scheduler_class(network, profile) # Make class object

    # Scheduling
    scheduler_object.do_schedule()

    # Visualization (Gantt chart?)

    
