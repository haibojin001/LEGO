import cytoolz as _cytoolz
from . import operator
from . import exceptions as _exceptions
from .exceptions import merge, merge_with
from cytoolz import (
    apply,
    comp,
    complement,
    compose,
    compose_left,
    concat,
    concatv,
    count,
    curry,
    diff,
    first,
    flip,
    frequencies,
    identity,
    interleave,
    isdistinct,
    isiterable,
    juxt,
    last,
    memoize,
    merge_sorted,
    peek,
    pipe,
    second,
    thread_first,
    thread_last,
)

_curried_names = (
    "accumulate",
    "assoc",
    "assoc_in",
    "cons",
    "countby",
    "dissoc",
    "do",
    "drop",
    "excepts",
    "filter",
    "get",
    "get_in",
    "groupby",
    "interpose",
    "itemfilter",
    "itemmap",
    "iterate",
    "join",
    "keyfilter",
    "keymap",
    "map",
    "mapcat",
    "nth",
    "partial",
    "partition",
    "partition_all",
    "partitionby",
    "peekn",
    "pluck",
    "random_sample",
    "reduce",
    "reduceby",
    "remove",
    "sliding_window",
    "sorted",
    "tail",
    "take",
    "take_nth",
    "topk",
    "unique",
    "update_in",
    "valfilter",
    "valmap",
)

for _name in _curried_names:
    globals()[_name] = _cytoolz.curry(getattr(_cytoolz, _name))

del _name
del _curried_names
del _exceptions
del _cytoolz