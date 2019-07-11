import unicodedata
import google.protobuf


def convert_to_str(uni):
    if isinstance(uni, str):
        return uni
    elif isinstance(uni, unicode):
        return unicodedata.normalize('NFKD', uni).encode('ascii', 'ignore')
    elif isinstance(uni, google.protobuf.pyext._message.RepeatedScalarContainer):
        return str(uni[0])
    else:
        return str(uni[0])
