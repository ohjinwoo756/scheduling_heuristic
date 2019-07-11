class Edge(object):
    def __init__(self, buffer_size, sender, receiver):
        self._buffer_size = buffer_size
        self._data_time = []
        self.sender = sender
        self.receiver = receiver

        # FIXME
        self.cpu2gpu = False
        self.gpu2cpu = False
        self.cpu2npu = False
        self.npu2cpu = False
        self.gpu_npu_connection = False
        self.transition_flag = False

    def do_init(self):
        del self._data_time
        self._data_time = []

        # FIXME
        self.cpu2gpu = False
        self.gpu2cpu = False
        self.cpu2npu = False
        self.npu2cpu = False
        self.gpu_npu_connection = False
        self.transition_flag = False

    def _num_data(self):
        return len(self._data_time)

    def check_in_buffer(self, time):
        if self._num_data() == 0:
            return False, False
        elif self._data_time[0] > time:
            return False, True
        else:
            return True, True

    def check_out_buffer(self):
        if self._num_data() == self._buffer_size:
            return False
        else:
            return True

    def produce_data(self, time):
        assert self._num_data() < self._buffer_size, "Buffer is full"
        self._data_time.append(time)
        self.receiver.need_in_edge_check = True

    def consume_data(self):
        assert self._num_data() > 0, "No input tocken"
        self._data_time.pop(0)
        self.sender.need_out_edge_check = True

    # FIXME
    def calc_transition_time(self):
        map_time = self.sender.get_map_time()
        unmap_time = self.sender.get_unmap_time()
        memcpy_time = self.sender.get_memcpy_time()
        # print("source: {} dest: {}".format(self.sender.pe.get_type(), self.receiver.pe.get_type()))
        if self.gpu2cpu:
            return map_time + memcpy_time
        elif self.cpu2gpu:
            return unmap_time + map_time + memcpy_time
        elif self.npu2cpu: # FIXME cpu2npu, npu2cpu time is same as cpu2gpu, gpu2cpu
            return map_time + memcpy_time
        elif self.cpu2npu:
            return unmap_time + map_time + memcpy_time
        elif self.gpu_npu_connection:
            return map_time + memcpy_time
        elif self.transition_flag:
            return memcpy_time
        else:
            return 0

    def __str__(self):
        return '{} -> {}'.format(self.sender.name, self.receiver.name)
