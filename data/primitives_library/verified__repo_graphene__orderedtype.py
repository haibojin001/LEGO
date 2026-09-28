from functools import total_ordering


@total_ordering
class OrderedType:
    creation_counter = 1

    def __init__(self, _creation_counter=None):
        if _creation_counter:
            self.creation_counter = _creation_counter
        else:
            self.creation_counter = self.gen_counter()

    @staticmethod
    def gen_counter():
        current = OrderedType.creation_counter
        OrderedType.creation_counter = current + 1
        return current

    def reset_counter(self):
        self.creation_counter = self.gen_counter()

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.creation_counter == other.creation_counter
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, OrderedType):
            return self.creation_counter < other.creation_counter
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, OrderedType):
            return self.creation_counter > other.creation_counter
        return NotImplemented

    def __hash__(self):
        return hash(self.creation_counter)