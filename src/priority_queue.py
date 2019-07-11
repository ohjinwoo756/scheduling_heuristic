import heapq
import itertools


class PriorityQueue:
    pq = []
    elements = {}
    counter = itertools.count()

    def __init__(self):
        self.pq = []
        self.elements = {}

    def insert(self, priority, value):
        if value in self.elements:
            return

        count = next(self.counter)
        entry = [priority, count, value]
        self.elements[value] = entry
        heapq.heappush(self.pq, entry)

    def delete(self, value):
        entry = self.elements.pop(value)
        entry[-1] = None

    def pop(self):
        while self.pq:
            priority, _, value = heapq.heappop(self.pq)
            if value != None:
                del self.elements[value]
                return priority, value
        raise KeyError('Pop from an empty PriorityQueue')

    def size(self):
        return len(self.elements)

    def print_queue(self):
        print("================= queue =================")
        for v in self.elements:
            print(v)
        print("=========================================")
