# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg863::array.array+random.random
# name: array_random_primitive
# summary: Uses array.array, random.random across 2 repos
# anchor_symbols: ['array.array', 'random.random']
# observed in 2 repos: ['JacksonWuxs__DaPy', 'rpy2__rpy2']...

# --- from JacksonWuxs__DaPy::DaPy/core/base/Matrix.py::Matrix.make_random ---
def make_random(cls, Ln, Col, type_int=False):
        if not (isinstance(Ln, int) and isinstance(Col, int)):
            raise TypeError("arguments `Ln` and `Col` expect <int> type,")
        if not isinstance(type_int, (bool, tuple)):
            raise TypeError("argutments `type_int` expects `False` symbol"+\
                            " or a tuple-like.")
        cls = cls()
        cls._matrix = [0] * Ln
        if type_int:
            for i in range(Ln):
                self._matrix[i] = array('f', [randint(*type_int)] * Col)
        else:
            for i in range(Ln):
                self._matrix[i] = array('f', [random()] * Col)
        cls._dim = Matrix.dims(Ln, Col)
        return cls

# --- from rpy2__rpy2::doc/_static/demos/benchmarks.py::setup_func ---
def setup_func(kind):
#-- setup_sum-begin
    n = 20000
    x_list = [random.random() for i in range(n)]
    module = None
    if kind == "array.array":
        import array as module
        res = module.array('f', x_list)
    elif kind == "numpy.array":
        import numpy as module
        res = module.array(x_list, 'f')
    elif kind == "FloatVector":
        import rpy2.robjects as module
        res = module.FloatVector(x_list)
    elif kind == "FloatSexpVector":
        import rpy2.rinterface as module
        module.initr()
        res = module.FloatSexpVector(x_list)
    elif kind == "FloatSexpVector-memoryview-array":
        import rpy2.rinterface as module
        module.initr()
        tmp = module.FloatSexpVector(x_list)
        mv = tmp.memoryview()
        res = array.array(mv.format, mv)
    elif kind == "list":
        res = x_list
    elif kind == "R":
        import rpy2.robjects as module
        res = module.rinterface.FloatSexpVector(x_list)
        module.globalenv['x'] = res
        res = None
#-- setup_sum-end
    else:
        raise ValueError("Unknown kind '%s'" %kind)
    return (res, module)
