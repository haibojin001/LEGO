import copy
import copyreg
import math
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from inspect import isclass
from operator import eq, lt

from . import tools

__type__ = object


class PrimitiveTree(list):
    def __init__(self, content):
        list.__init__(self, content)

    def __deepcopy__(self, memo):
        duplicate = self.__class__(self)
        duplicate.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return duplicate

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError(
                    "Invalid slice object (try to assign a %s in a tree of size %d). "
                    "Even if this is allowed by the list object slice setter, this "
                    "should not be done in the PrimitiveTree context, as this may "
                    "lead to an unpredictable behavior for searchSubtree or evaluate."
                    % (key, len(self))
                )
            total = value[0].arity
            for element in value[1:]:
                total += element.arity - 1
            if total != 0:
                raise ValueError(
                    "Invalid slice assignation : insertion of an incomplete subtree "
                    "is not allowed in PrimitiveTree. A tree is defined as incomplete "
                    "when some nodes cannot be mapped to any position in the tree, "
                    "considering the primitives' arity."
                )
        elif value.arity != self[key].arity:
            raise ValueError(
                "Invalid node replacement with a node of a different arity."
            )
        list.__setitem__(self, key, value)

    def __str__(self):
        expression = ""
        stack = []
        for node in self:
            stack.append((node, []))
            while stack and len(stack[-1][1]) == stack[-1][0].arity:
                primitive, arguments = stack.pop()
                expression = primitive.format(*arguments)
                if not stack:
                    break
                stack[-1][1].append(expression)
        return expression

    @classmethod
    def from_string(cls, string, pset):
        tokens = re.split(r"[ \t\n\r\f\v(),]", string)
        expression = []
        pending_types = deque()

        for token in tokens:
            if not token:
                continue

            expected = pending_types.popleft() if pending_types else None
            if token in pset.mapping:
                item = pset.mapping[token]
                if expected is not None and not issubclass(item.ret, expected):
                    raise TypeError(
                        "Primitive {} return type {} does not match the expected one: {}."
                        .format(item, item.ret, expected)
                    )
                expression.append(item)
                if isinstance(item, Primitive):
                    pending_types.extendleft(reversed(item.args))
            else:
                try:
                    value = eval(token)
                except NameError:
                    raise TypeError("Unable to evaluate terminal: {}.".format(token))

                if expected is None:
                    expected = type(value)
                if not issubclass(type(value), expected):
                    raise TypeError(
                        "Terminal {} type {} does not match the expected one: {}."
                        .format(value, type(value), expected)
                    )
                expression.append(Terminal(value, False, expected))

        return cls(expression)

    @property
    def height(self):
        depths = [0]
        maximum = 0
        for node in self:
            depth = depths.pop()
            maximum = max(maximum, depth)
            depths.extend([depth + 1] * node.arity)
        return maximum

    @property
    def root(self):
        return self[0]

    def searchSubtree(self, begin):
        end = begin + 1
        remaining = self[begin].arity
        while remaining > 0:
            remaining += self[end].arity - 1
            end += 1
        return slice(begin, end)


class Primitive:
    __slots__ = ("name", "arity", "args", "ret", "seq")

    def __init__(self, name, args, ret):
        self.name = name
        self.arity = len(args)
        self.args = args
        self.ret = ret
        fields = ", ".join("{{{}}}".format(index) for index in range(self.arity))
        self.seq = "{}({})".format(name, fields)

    def format(self, *args):
        return self.seq.format(*args)

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in self.__slots__)


class Terminal:
    __slots__ = ("name", "value", "ret", "conv_fct")

    def __init__(self, terminal, symbolic, ret):
        self.ret = ret
        self.value = terminal
        self.name = str(terminal)
        self.conv_fct = str if symbolic else repr

    @property
    def arity(self):
        return 0

    def format(self):
        return self.conv_fct(self.value)

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in self.__slots__)


class MetaEphemeral(type):
    cache = {}

    def __new__(meta, name, func, ret=__type__, id_=None):
        if id_ in meta.cache:
            return meta.cache[id_]

        if isinstance(func, types.LambdaType) and func.__name__ == "<lambda>":
            warnings.warn(
                "Ephemeral {} function cannot be pickled because its generating "
                "function is a lambda function. Use functools.partial instead."
                .format(name),
                RuntimeWarning,
            )

        def __init__(self):
            self.value = func()

        attributes = {
            "__init__": __init__,
            "name": name,
            "func": staticmethod(func),
            "ret": ret,
            "conv_fct": repr,
        }
        cls = type.__new__(meta, name, (Terminal,), attributes)
        meta.cache[id(cls)] = cls
        return cls

    def __reduce__(cls):
        return (MetaEphemeral, (cls.name, cls.func, cls.ret, id(cls)))


copyreg.pickle(MetaEphemeral, MetaEphemeral.__reduce__)


class Ephemeral(Terminal, metaclass=MetaEphemeral):
    pass


class PrimitiveSetTyped:
    def __init__(self, name, in_types, ret_type, prefix="ARG"):
        self.name = name
        self.ins = list(in_types)
        self.ret = ret_type
        self.prim_dict = defaultdict(list)
        self.term_dict = defaultdict(list)
        self.context = {"__builtins__": None}
        self.mapping = {}
        self.terms_count = 0
        self.prims_count = 0

        self.arguments = ["{}{}".format(prefix, index) for index in range(len(in_types))]
        for argument, type_ in zip(self.arguments, in_types):
            terminal = Terminal(argument, True, type_)
            self._add_terminal(terminal)
            self.mapping[argument] = terminal

    def _add(self, item, dictionary):
        def add_type(type_):
            if type_ not in dictionary:
                dictionary[type_] = []
                for registered_type, items in dictionary.items():
                    if registered_type is not type_ and issubclass(registered_type, type_):
                        dictionary[type_].extend(items)

        add_type(item.ret)
        if isinstance(item, Primitive):
            for arg_type in item.args:
                add_type(arg_type)

        for type_ in dictionary:
            if issubclass(item.ret, type_):
                dictionary[type_].append(item)

    def _add_primitive(self, primitive):
        self._add(primitive, self.prim_dict)

    def _add_terminal(self, terminal):
        self._add(terminal, self.term_dict)

    def addPrimitive(self, primitive, in_types, ret_type, name=None):
        if name is None:
            name = primitive.__name__
        prim = Primitive(name, in_types, ret_type)
        self._add_primitive(prim)
        self.context[name] = primitive
        self.mapping[name] = prim
        self.prims_count += 1

    def addTerminal(self, terminal, ret_type, name=None):
        symbolic = name is not None
        if name is None and isclass(terminal):
            name = terminal.__name__
            symbolic = True
        elif name is None:
            name = terminal

        term = Terminal(name, symbolic, ret_type)
        self._add_terminal(term)
        self.mapping[term.name] = term

        if symbolic:
            self.context[term.name] = terminal
        self.terms_count += 1

    def addEphemeralConstant(self, name, ephemeral, ret_type):
        if name not in self.mapping:
            class_ = MetaEphemeral(name, ephemeral, ret_type)
        else:
            class_ = self.mapping[name]
            if class_.func is not ephemeral:
                raise Exception(
                    "Ephemerals with different functions should be named differently, "
                    "even between psets."
                )
            if class_.ret is not ret_type:
                raise Exception(
                    "Ephemerals with the same name and function should have the same "
                    "type, even between psets."
                )

        self._add_terminal(class_)
        self.mapping[name] = class_
        self.terms_count += 1

    def addADF(self, adfset):
        self.addPrimitive(adfset.name, adfset.ins, adfset.ret)

    def renameArguments(self, **kargs):
        for i, old_name in enumerate(self.arguments):
            if old_name in kargs:
                new_name = kargs[old_name]
                self.arguments[i] = new_name
                self.mapping[new_name] = self.mapping[old_name]
                self.mapping[new_name].value = new_name
                self.mapping[new_name].name = new_name
                del self.mapping[old_name]

    @property
    def terminalRatio(self):
        return self.terms_count / float(self.terms_count + self.prims_count)


class PrimitiveSet(PrimitiveSetTyped):
    def __init__(self, name, arity, prefix="ARG"):
        super().__init__(name, [__type__] * arity, __type__, prefix)

    def addPrimitive(self, primitive, arity, name=None):
        super().addPrimitive(primitive, [__type__] * arity, __type__, name)

    def addTerminal(self, terminal, name=None):
        super().addTerminal(terminal, __type__, name)

    def addEphemeralConstant(self, name, ephemeral):
        super().addEphemeralConstant(name, ephemeral, __type__)


def compile(expr, pset):
    code = str(expr)
    if len(pset.arguments) > 0:
        code = "lambda {}: {}".format(",".join(pset.arguments), code)

    try:
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError(
            "DEAP : Error in tree evaluation : Python cannot evaluate a tree "
            "higher than 90. To avoid this, you should use bloat control on your "
            "operators. See the DEAP documentation for more information."
        ).with_traceback(traceback)


def compileADF(expr, psets):
    adfdict = {}
    for tree, pset in reversed(list(zip(expr, psets))):
        pset.context.update(adfdict)
        adfdict[pset.name] = compile(tree, pset)
    return adfdict[psets[0].name]


def genFull(pset, min_, max_, type_=None):
    return generate(pset, min_, max_, lambda height, depth: depth == height, type_)


def genGrow(pset, min_, max_, type_=None):
    def condition(height, depth):
        return depth == height or (
            depth >= min_ and random.random() < pset.terminalRatio
        )

    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


def generate(pset, min_, max_, condition, type_=None):
    if type_ is None:
        type_ = pset.ret
    height = random.randint(min_, max_)
    expression = []
    stack = [(0, type_)]

    while stack:
        depth, expected = stack.pop()
        if condition(height, depth):
            try:
                terminal = random.choice(pset.term_dict[expected])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add a terminal of type "
                    "'{}', but there is none available.".format(expected)
                ).with_traceback(traceback)

            if type(terminal) is MetaEphemeral:
                terminal = terminal()
            expression.append(terminal)
        else:
            try:
                primitive = random.choice(pset.prim_dict[expected])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add a primitive of type "
                    "'{}', but there is none available.".format(expected)
                ).with_traceback(traceback)

            expression.append(primitive)
            stack.extend(reversed([(depth + 1, arg) for arg in primitive.args]))

    return expression


def cxOnePoint(ind1, ind2):
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    types1 = defaultdict(list)
    types2 = defaultdict(list)

    if ind1.root.ret == ind2.root.ret:
        types1[ind1.root.ret].append(0)
        types2[ind2.root.ret].append(0)

    for index, node in enumerate(ind1[1:], 1):
        types1[node.ret].append(index)
    for index, node in enumerate(ind2[1:], 1):
        types2[node.ret].append(index)

    common = set(types1).intersection(types2)
    if common:
        type_ = random.choice(list(common))
        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])
        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def cxOnePointLeafBiased(ind1, ind2, termpb):
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    choose_terminal = random.random() < termpb
    predicate = eq if choose_terminal else lt
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for index, node in enumerate(ind1[1:], 1):
        if predicate(node.arity, 0):
            types1[node.ret].append(index)
    for index, node in enumerate(ind2[1:], 1):
        if predicate(node.arity, 0):
            types2[node.ret].append(index)

    common = set(types1).intersection(types2)
    if common:
        type_ = random.choice(list(common))
        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])
        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def mutUniform(individual, expr, pset):
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual,


def mutNodeReplacement(individual, pset):
    if len(individual) < 2:
        return individual,

    index = random.randrange(1, len(individual))
    node = individual[index]

    if node.arity == 0:
        replacement = random.choice(pset.term_dict[node.ret])
        if type(replacement) is MetaEphemeral:
            replacement = replacement()
        individual[index] = replacement
    else:
        candidates = [
            primitive
            for primitive in pset.prim_dict[node.ret]
            if primitive.args == node.args
        ]
        individual[index] = random.choice(candidates)

    return individual,


def mutEphemeral(individual, mode):
    if mode not in ("one", "all"):
        raise ValueError("Mode must be one of 'one' or 'all'.")

    indices = [
        index
        for index, node in enumerate(individual)
        if isinstance(type(node), MetaEphemeral)
    ]

    if mode == "one":
        if indices:
            index = random.choice(indices)
            individual[index] = type(individual[index])()
    else:
        for index in indices:
            individual[index] = type(individual[index])()

    return individual,


def mutInsert(individual, pset):
    index = random.randrange(len(individual))
    node = individual[index]
    primitives = []

    for primitive in pset.prim_dict[node.ret]:
        if node.ret in primitive.args:
            primitives.append(primitive)

    if primitives:
        primitive = random.choice(primitives)
        argument = random.choice(
            [position for position, arg_type in enumerate(primitive.args) if arg_type == node.ret]
        )
        subtree = [primitive]
        for position, arg_type in enumerate(primitive.args):
            if position == argument:
                subtree.append(node)
            else:
                terminal = random.choice(pset.term_dict[arg_type])
                subtree.append(terminal() if type(terminal) is MetaEphemeral else terminal)

        individual[individual.searchSubtree(index)] = subtree

    return individual,


def mutShrink(individual):
    if len(individual) < 3:
        return individual,

    candidates = []
    for index, node in enumerate(individual):
        if node.arity > 0:
            slice_ = individual.searchSubtree(index)
            for child in range(index + 1, slice_.stop):
                if individual[child].ret == node.ret:
                    candidates.append((index, child))

    if candidates:
        parent, child = random.choice(candidates)
        child_slice = individual.searchSubtree(child)
        individual[individual.searchSubtree(parent)] = individual[child_slice]

    return individual,


def staticLimit(key, max_value):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            originals = [copy.deepcopy(arg) for arg in args]
            offspring = list(function(*args, **kwargs))
            for index, individual in enumerate(offspring):
                if key(individual) > max_value:
                    offspring[index] = random.choice(originals)
            return offspring
        return wrapper
    return decorator


def harm(individual, toolbox, cxpb, mutpb, ngen, alpha, beta, gamma, rho,
         nbrindsmodel=-1, mincutoff=20, stats=None, halloffame=None,
         verbose=__debug__):
    def halflife(x):
        return x * float(alpha) / 2.0

    def cutoffsize(population):
        sizes = sorted(len(ind) for ind in population)
        if not sizes:
            return mincutoff
        if len(sizes) < 2:
            return max(mincutoff, sizes[0])

        mean = sum(sizes) / float(len(sizes))
        variance = sum((size - mean) ** 2 for size in sizes) / float(len(sizes))
        return max(mincutoff, int(mean + beta * math.sqrt(variance)))

    def probability(size, cutoff):
        if size <= cutoff:
            return 1.0
        return (1.0 + gamma * (size - cutoff)) ** (-rho)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    population = [toolbox.clone(individual)]
    population.extend(toolbox.population(n=nbrindsmodel if nbrindsmodel > 0 else 1))
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    for generation in range(1, ngen + 1):
        cutoff = cutoffsize(population)
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for first, second in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                first, second = toolbox.mate(first, second)
                del first.fitness.values
                del second.fitness.values

        for index, mutant in enumerate(offspring):
            if random.random() < mutpb:
                offspring[index], = toolbox.mutate(mutant)
                del offspring[index].fitness.values

        accepted = []
        for child in offspring:
            if random.random() <= probability(len(child), cutoff):
                accepted.append(child)

        invalid = [ind for ind in accepted if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid)
        for ind, fitness in zip(invalid, fitnesses):
            ind.fitness.values = fitness

        population = accepted
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=generation, nevals=len(invalid), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def graph(expr):
    nodes = list(range(len(expr)))
    edges = []
    labels = {}

    stack = []
    for index, node in enumerate(expr):
        labels[index] = node.name
        if stack:
            parent, remaining = stack[-1]
            edges.append((parent, index))
            stack[-1] = (parent, remaining - 1)
            while stack and stack[-1][1] == 0:
                stack.pop()
        if node.arity > 0:
            stack.append((index, node.arity))

    return nodes, edges, labels