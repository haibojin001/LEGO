import array
import copy
import copyreg
import warnings

class_replacers = {}

try:
    import numpy
    _numpy_types = (numpy.ndarray, numpy.array)
except ImportError:
    pass
except AttributeError:
    pass
else:
    class _numpy_array(numpy.ndarray):
        @staticmethod
        def __new__(cls, iterable):
            return numpy.array(list(iterable)).view(cls)

        def __deepcopy__(self, memo):
            result = numpy.ndarray.copy(self)
            result.__dict__.update(copy.deepcopy(self.__dict__, memo))
            return result

        def __setstate__(self, state):
            self.__dict__.update(state)

        def __reduce__(self):
            return self.__class__, (list(self),), self.__dict__

    class_replacers[numpy.ndarray] = _numpy_array


class _array(array.array):
    @staticmethod
    def __new__(cls, seq=()):
        return super(_array, cls).__new__(cls, cls.typecode, seq)

    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__, self)
        memo[id(self)] = result
        result.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self):
        return self.__class__, (list(self),), self.__dict__


class_replacers[array.array] = _array


class MetaCreator(type):
    def __new__(mcls, name, base, namespace):
        return super(MetaCreator, mcls).__new__(mcls, name, (base,), namespace)

    def __init__(cls, name, base, namespace):
        instance_members = {}
        class_members = {}

        for key, value in namespace.items():
            if isinstance(value, type):
                instance_members[key] = value
            else:
                class_members[key] = value

        def initialize(self, *args, **kwargs):
            for key, value_type in instance_members.items():
                setattr(self, key, value_type())
            if base.__init__ is not object.__init__:
                base.__init__(self, *args, **kwargs)

        cls.__init__ = initialize
        cls.reduce_args = (name, base, namespace)
        super(MetaCreator, cls).__init__(name, (base,), class_members)

    def __reduce__(cls):
        return meta_create, cls.reduce_args


copyreg.pickle(MetaCreator, MetaCreator.__reduce__)


def meta_create(name, base, dct):
    created = MetaCreator(name, base, dct)
    globals()[name] = created
    return created


def create(name, base, **kargs):
    if name in globals():
        warnings.warn(
            "A class named '{0}' has already been created and it "
            "will be overwritten. Consider deleting previous "
            "creation of that class or rename it.".format(name),
            RuntimeWarning,
        )

    replacement = class_replacers.get(base)
    if replacement is not None:
        base = replacement

    meta_create(name, base, kargs)