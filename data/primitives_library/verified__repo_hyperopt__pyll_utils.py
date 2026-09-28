from functools import partial, wraps

from .base import DuplicateLabel
from .pyll import as_apply, scope
from .pyll.base import Apply, Literal, MissingArgument


def validate_label(function):
    @wraps(function)
    def checked(label, *args, **kwargs):
        direct_string = isinstance(label, (str, bytes))
        literal_string = isinstance(label, Literal) and isinstance(
            label.obj, (str, bytes)
        )
        if not (direct_string or literal_string):
            raise TypeError("require string label")
        return function(label, *args, **kwargs)

    return checked


def validate_distribution_range(function):
    @wraps(function)
    def checked(label, *args, **kwargs):
        low = args[0] if args else kwargs.get("low")
        high = args[1] if len(args) > 1 else kwargs.get("high")

        if low and high and not low < high:
            raise ValueError(
                "low should be less than high: %s is not smaller than %s"
                % (low, high)
            )
        return function(label, *args, **kwargs)

    return checked


@scope.define
def hyperopt_param(label, obj):
    """Annotate a stochastic expression as a hyperparameter."""
    return obj


@validate_label
def hp_pchoice(label, p_options):
    probabilities, options = zip(*p_options)
    choice = scope.hyperopt_param(label, scope.categorical(probabilities))
    return scope.switch(choice, *options)


@validate_label
def hp_choice(label, options):
    choice = scope.hyperopt_param(label, scope.randint(len(options)))
    return scope.switch(choice, *options)


@validate_label
def hp_randint(label, *args, **kwargs):
    expression = scope.hyperopt_param(label, scope.randint(*args, **kwargs))
    return scope.int(expression)


@validate_label
@validate_distribution_range
def hp_uniform(label, *args, **kwargs):
    expression = scope.hyperopt_param(label, scope.uniform(*args, **kwargs))
    return scope.float(expression)


@validate_label
def hp_uniformint(label, *args, **kwargs):
    kwargs["q"] = 1.0
    return scope.int(hp_quniform(label, *args, **kwargs))


@validate_label
@validate_distribution_range
def hp_quniform(label, *args, **kwargs):
    expression = scope.hyperopt_param(label, scope.quniform(*args, **kwargs))
    return scope.float(expression)


@validate_label
@validate_distribution_range
def hp_loguniform(label, *args, **kwargs):
    expression = scope.hyperopt_param(label, scope.loguniform(*args, **kwargs))
    return scope.float(expression)


@validate_label
@validate_distribution_range
def hp_qloguniform(label, *args, **kwargs):
    expression = scope.hyperopt_param(label, scope.qloguniform(*args, **kwargs))
    return scope.float(expression)


@validate_label
def hp_normal(label, *args, **kwargs):
    expression = scope.hyperopt_param(label, scope.normal(*args, **kwargs))
    return scope.float(expression)


@validate_label
def hp_qnormal(label, *args, **kwargs):
    expression = scope.hyperopt_param(label, scope.qnormal(*args, **kwargs))
    return scope.float(expression)


@validate_label
def hp_lognormal(label, *args, **kwargs):
    expression = scope.hyperopt_param(label, scope.lognormal(*args, **kwargs))
    return scope.float(expression)


@validate_label
def hp_qlognormal(label, *args, **kwargs):
    expression = scope.hyperopt_param(label, scope.qlognormal(*args, **kwargs))
    return scope.float(expression)


class Cond:
    def __init__(self, name, val, op):
        self.op = op
        self.name = name
        self.val = val

    def __str__(self):
        return "Cond{%s %s %s}" % (self.name, self.op, self.val)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            self.op == other.op
            and self.name == other.name
            and self.val == other.val
        )

    def __hash__(self):
        return hash((self.op, self.name, self.val))


EQ = partial(Cond, op="=")


def _expr_to_config(expr, conditions, hps):
    if expr.name == "switch":
        selector = expr.inputs()[0]
        choices = expr.inputs()[1:]

        assert selector.name == "hyperopt_param"
        assert selector.arg["obj"].name in ("randint", "categorical")

        _expr_to_config(selector, conditions, hps)

        label = selector.arg["label"].obj
        for index, choice in enumerate(choices):
            _expr_to_config(choice, conditions + (EQ(label, index),), hps)
        return

    if expr.name == "hyperopt_param":
        label = expr.arg["label"].obj
        distribution = expr.arg["obj"]

        if label in hps:
            if hps[label]["node"] != distribution:
                raise DuplicateLabel(label)
            hps[label]["conditions"].add(conditions)
        else:
            hps[label] = {
                "node": distribution,
                "conditions": {conditions},
                "label": label,
            }
        return

    for child in expr.inputs():
        _expr_to_config(child, conditions, hps)


def expr_to_config(expr, conditions, hps):
    """Extract hyperparameter nodes and their activation conditions."""
    expression = as_apply(expr)
    if conditions is None:
        conditions = ()

    assert isinstance(expression, Apply)
    _expr_to_config(expression, conditions, hps)
    _remove_allpaths(hps, conditions)


def _remove_allpaths(hps, conditions):
    possible_conditions = {}

    for label, info in list(hps.items()):
        node = info["node"]

        if node.name == "randint":
            low = node.arg["low"].obj
            high = node.arg["high"]
            size = high.obj - low if high != MissingArgument else low
            possible_conditions[label] = frozenset(
                EQ(label, value) for value in range(size)
            )

        elif node.name == "categorical":
            probabilities = node.arg["p"].obj
            possible_conditions[label] = frozenset(
                EQ(label, value) for value in range(probabilities.size)
            )

    for label, info in list(hps.items()):
        if len(info["conditions"]) <= 1:
            continue

        paths = [
            [condition for condition in path if condition is not True]
            for path in info["conditions"]
        ]
        paths = [path for path in paths if path]

        if not paths:
            info["conditions"] = {conditions}
            continue

        dependency = paths[0][0].name
        is_single_dependency = all(
            len(path) == 1 and path[0].name == dependency for path in paths
        )

        if is_single_dependency:
            found = [path[0] for path in paths]
            if frozenset(found) == possible_conditions[dependency]:
                info["conditions"] = {conditions}