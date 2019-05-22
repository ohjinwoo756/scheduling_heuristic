
import caffe_pb2
import google.protobuf.text_format

def read_network_prototxt(network_prototxt_path):

    network = caffe_pb2.NetParameter() # load NetParameter
    fp = open(network_prototxt_path, 'r')
    google.protobuf.text_format.Merge(fp.read(), network) # Merge read prototxt with NetParameter => store to 'network'
    fp.close()
    return network


def read_network_profile_prototxt(network_profile_prototxt_path):

    profile = caffe_pb2.NetParameter() # load NetParameter
    fp = open(network_profile_prototxt_path, 'r')
    google.protobuf.text_format.Merge(fp.read(), profile) # Merge read prototxt with NetParameter => store to 'profile'
    fp.close()
    return profile
