from enum import Enum, auto
from functools import wraps
from types import FunctionType
from typing import TYPE_CHECKING, Literal, Union

import ray

from ...protocol import DataProto, DataProtoFuture

if TYPE_CHECKING:
    from .worker_group import WorkerGroup


MAGIC_ATTR = "attrs_3141562937"


class Dispatch(Enum):
    RANK_ZERO = auto()
    ONE_TO_ALL = auto()
    ALL_TO_ALL = auto()
    DP_COMPUTE = auto()
    DP_COMPUTE_PROTO = auto()
    DP_COMPUTE_PROTO_WITH_FUNC = auto()
    DP_COMPUTE_METRIC = auto()


class Execute(Enum):
    ALL = 0
    RANK_ZERO = 1


def _split_args_kwargs_data_proto(chunks: int, *args, **kwargs):
    split_args = []
    for value in args:
        assert isinstance(value, (DataProto, DataProtoFuture))
        split_args.append(value.chunk(chunks=chunks))

    split_kwargs = {}
    for name, value in kwargs.items():
        assert isinstance(value, (DataProto, DataProtoFuture))
        split_kwargs[name] = value.chunk(chunks=chunks)

    return split_args, split_kwargs


def dispatch_one_to_all(worker_group: "WorkerGroup", *args, **kwargs):
    dispatched_args = tuple([value] * worker_group.world_size for value in args)
    dispatched_kwargs = {name: [value] * worker_group.world_size for name, value in kwargs.items()}
    return dispatched_args, dispatched_kwargs


def dispatch_all_to_all(worker_group: "WorkerGroup", *args, **kwargs):
    return args, kwargs


def collect_all_to_all(worker_group: "WorkerGroup", output):
    return output


def _concat_data_proto_or_future(outputs: list[DataProto]) -> DataProto:
    for value in outputs:
        assert type(value) is type(outputs[0])

    first = outputs[0]
    if isinstance(first, DataProto):
        return DataProto.concat(outputs)
    if isinstance(first, ray.ObjectRef):
        return DataProtoFuture.concat(outputs)
    raise NotImplementedError


def dispatch_dp_compute(worker_group: "WorkerGroup", *args, **kwargs):
    for value in args:
        assert isinstance(value, (tuple, list)) and len(value) == worker_group.world_size

    for value in kwargs.values():
        assert isinstance(value, (tuple, list)) and len(value) == worker_group.world_size

    return args, kwargs


def collect_dp_compute(worker_group: "WorkerGroup", outputs: list[DataProto]) -> list[DataProto]:
    assert len(outputs) == worker_group.world_size
    return outputs


def dispatch_dp_compute_data_proto(worker_group: "WorkerGroup", *args, **kwargs):
    split_args, split_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args, **kwargs)
    return split_args, split_kwargs


def dispatch_dp_compute_data_proto_with_func(worker_group: "WorkerGroup", *args, **kwargs):
    assert type(args[0]) is FunctionType
    split_args, split_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args[1:], **kwargs)
    split_args_with_func = [[args[0]] * worker_group.world_size] + split_args
    return split_args_with_func, split_kwargs


def collect_dp_compute_data_proto(worker_group: "WorkerGroup", outputs: list[DataProto]) -> DataProto:
    for value in outputs:
        assert isinstance(value, (DataProto, ray.ObjectRef)), f"Expect a DataProto, but got {type(value)}"

    outputs = collect_dp_compute(worker_group, outputs)
    return _concat_data_proto_or_future(outputs)


def get_predefined_dispatch_fn(dispatch_mode: Dispatch):
    modes = {
        Dispatch.ONE_TO_ALL: {
            "dispatch_fn": dispatch_one_to_all,
            "collect_fn": collect_all_to_all,
        },
        Dispatch.ALL_TO_ALL: {
            "dispatch_fn": dispatch_all_to_all,
            "collect_fn": collect_all_to_all,
        },
        Dispatch.DP_COMPUTE: {
            "dispatch_fn": dispatch_dp_compute,
            "collect_fn": collect_dp_compute,
        },
        Dispatch.DP_COMPUTE_PROTO: {
            "dispatch_fn": dispatch_dp_compute_data_proto,
            "collect_fn": collect_dp_compute_data_proto,
        },
        Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
            "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
            "collect_fn": collect_dp_compute_data_proto,
        },
        Dispatch.DP_COMPUTE_METRIC: {
            "dispatch_fn": dispatch_dp_compute_data_proto,
            "collect_fn": collect_dp_compute,
        },
    }
    return modes[dispatch_mode]


def get_predefined_execute_fn(execute_mode: Execute):
    modes = {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }
    return modes[execute_mode]


def _check_dispatch_mode(dispatch_mode: Union[Dispatch, dict[Literal["dispatch_fn", "collect_fn"], FunctionType]]):
    assert isinstance(dispatch_mode, (Dispatch, dict)), (
        f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    )
    if isinstance(dispatch_mode, dict):
        for key in ["dispatch_fn", "collect_fn"]:
            assert key in dispatch_mode, f"key {key} should be in dispatch_mode if it is a dictionary"


def _check_execute_mode(execute_mode: Execute):
    assert isinstance(execute_mode, Execute), f"execute_mode must be a Execute. Got {execute_mode}"


def _materialize_futures(*args, **kwargs):
    materialized_args = []
    for value in args:
        if isinstance(value, DataProtoFuture):
            value = value.get()
        materialized_args.append(value)

    for name, value in kwargs.items():
        if isinstance(value, DataProtoFuture):
            kwargs[name] = value.get()

    return tuple(materialized_args), kwargs


def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        setattr(
            inner,
            MAGIC_ATTR,
            {
                "dispatch_mode": dispatch_mode,
                "execute_mode": execute_mode,
                "blocking": blocking,
            },
        )
        return inner

    return decorator