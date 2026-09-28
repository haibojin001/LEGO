from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import eq


def identity(obj):
    return obj


class History(object):
    def __init__(self):
        self.genealogy_index = 0
        self.genealogy_history = {}
        self.genealogy_tree = {}

    def update(self, individuals):
        try:
            parent_indices = tuple(ind.history_index for ind in individuals)
        except AttributeError:
            parent_indices = tuple()

        for individual in individuals:
            self.genealogy_index += 1
            individual.history_index = self.genealogy_index
            self.genealogy_history[self.genealogy_index] = deepcopy(individual)
            self.genealogy_tree[self.genealogy_index] = parent_indices

    @property
    def decorator(self):
        def decorate(function):
            def wrapper(*args, **kwargs):
                individuals = function(*args, **kwargs)
                self.update(individuals)
                return individuals
            return wrapper
        return decorate

    def getGenealogy(self, individual, max_depth=float("inf")):
        tree = {}
        visited = set()

        def visit(index, depth):
            if index not in self.genealogy_tree:
                return
            depth += 1
            if depth > max_depth:
                return

            parents = self.genealogy_tree[index]
            tree[index] = parents
            for parent in parents:
                if parent not in visited:
                    visit(parent, depth)
                visited.add(parent)

        visit(individual.history_index, 0)
        return tree


class Statistics(object):
    def __init__(self, key=identity):
        self.key = key
        self.functions = {}
        self.fields = []

    def register(self, name, function, *args, **kargs):
        self.functions[name] = partial(function, *args, **kargs)
        self.fields.append(name)

    def compile(self, data):
        values = tuple(self.key(element) for element in data)
        return {name: function(values) for name, function in self.functions.items()}


class MultiStatistics(dict):
    def __init__(self, **kwargs):
        super(MultiStatistics, self).__init__(kwargs)

    @property
    def fields(self):
        return sorted(self.keys())

    def compile(self, data):
        return {name: statistics.compile(data)
                for name, statistics in self.items()}

    def register(self, name, function, *args, **kargs):
        for statistics in self.values():
            statistics.register(name, function, *args, **kargs)


class Logbook(list):
    def __init__(self):
        super(Logbook, self).__init__()
        self.buffindex = 0
        self.chapters = defaultdict(Logbook)
        self.header = []
        self.log_header = True

    def record(self, **infos):
        common = {
            key: value for key, value in infos.items()
            if not isinstance(value, dict)
        }

        for key, value in infos.items():
            if isinstance(value, dict):
                chapter_infos = common.copy()
                chapter_infos.update(value)
                self.chapters[key].record(**chapter_infos)

        self.append(infos)

    def select(self, *names):
        if len(names) == 1:
            name = names[0]
            return [entry.get(name, None) for entry in self]
        return tuple([entry.get(name, None) for entry in self] for name in names)

    def __delitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            for index in reversed(list(indices)):
                self.pop(index)
        else:
            self.pop(key)

    def pop(self, index=0):
        if index < 0:
            index += len(self)

        for chapter in self.chapters.values():
            chapter.pop(index)

        return super(Logbook, self).pop(index)

    def _column_data(self, name, startindex, parent_entries=None):
        if parent_entries is None:
            parent_entries = self[startindex:]

        if name in self.chapters:
            chapter = self.chapters[name]
            chapter_entries = chapter[startindex:]
            return chapter._table_columns(0, chapter_entries)

        return [(name, [entry.get(name, "") for entry in parent_entries])]

    def _table_columns(self, startindex=0, entries=None):
        if entries is None:
            entries = self[startindex:]

        if self.header:
            names = list(self.header)
        elif entries:
            names = sorted(entries[0].keys()) + sorted(self.chapters.keys())
        else:
            names = sorted(self.chapters.keys())

        columns = []
        for name in names:
            columns.extend(self._column_data(name, startindex, entries))
        return columns

    def __txt__(self, startindex):
        if not self:
            return ""

        columns = self._table_columns(startindex)
        if not columns:
            return ""

        labels = [label for label, values in columns]
        values = [values for label, values in columns]

        lines = []
        if self.log_header and startindex == 0:
            lines.append(labels)

        row_count = max((len(column) for column in values), default=0)
        for row in range(row_count):
            lines.append([
                column[row] if row < len(column) else ""
                for column in values
            ])

        string_lines = [
            [str(value) for value in line]
            for line in lines
        ]
        widths = [
            max(len(line[column]) for line in string_lines)
            for column in range(len(columns))
        ]

        formatted = []
        for line in string_lines:
            formatted.append("\t".join(
                value.ljust(width)
                for value, width in zip(line, widths)
            ).rstrip())
        return "\n".join(formatted)

    @property
    def stream(self):
        startindex = self.buffindex
        self.buffindex = len(self)
        return self.__txt__(startindex)

    def __str__(self):
        return self.__txt__(0)


class HallOfFame(object):
    def __init__(self, maxsize, similar=eq):
        self.maxsize = maxsize
        self.keys = []
        self.items = []
        self.similar = similar

    def update(self, population):
        for individual in population:
            if len(self) == 0 and self.maxsize != 0:
                self.insert(individual)
                continue

            if individual.fitness > self[-1].fitness or len(self) < self.maxsize:
                for member in self:
                    if self.similar(individual, member):
                        break
                else:
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(individual)

    def insert(self, item):
        item = deepcopy(item)
        index = bisect_right(self.keys, item.fitness)
        self.items.insert(len(self) - index, item)
        self.keys.insert(index, item.fitness)

    def remove(self, index):
        del self.keys[len(self) - (index + 1)]
        del self.items[index]

    def clear(self):
        del self.keys[:]
        del self.items[:]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __str__(self):
        return str(self.items)


class ParetoFront(HallOfFame):
    def __init__(self, similar=eq):
        super(ParetoFront, self).__init__(float("inf"), similar)

    def update(self, population):
        for individual in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []

            for index, member in enumerate(self):
                if not dominates_one and member.fitness.dominates(individual.fitness):
                    is_dominated = True
                    break
                elif individual.fitness.dominates(member.fitness):
                    dominates_one = True
                    to_remove.append(index)
                elif individual.fitness == member.fitness and self.similar(individual, member):
                    has_twin = True
                    break

            if not is_dominated and not has_twin:
                for index in reversed(to_remove):
                    self.remove(index)
                self.insert(individual)