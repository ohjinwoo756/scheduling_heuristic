
def parse_options():

    # Create ArgumentParser
    from argparse import ArgumentParser
    parser = ArgumentParser(usage="[USAGE] ...")

    # Add argument
    parser.add_argument("-s", "--scheduler", dest='scheduler', help="Scheduler")
    parser.add_argument("-n", "--network", nargs='+', dest='network', help="Network to be scheduled") # nargs='+' : return list format for multiple applications
    parser.add_argument("-p", "--profile", nargs='+', dest='profile', help="Profiling of network") # nargs='+' : return list format for multiple applications

    # Parse argument
    args = parser.parse_args()

    return args
