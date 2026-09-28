import gc
from collections import defaultdict
from functools import partial
from typing import Callable, Union

import torch
import torch.distributed.fsdp._traversal_utils as _traversal_utils
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from torch.optim import Optimizer
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name


def get_init_fn(model: nn.Module, device: Union[str, torch.device]) -> Callable[[nn.Module], None]:
    occurrence_count = defaultdict(int)
    for _, parameter in model.named_parameters(remove_duplicate=False):
        occurrence_count[parameter] += 1

    shared_parameters = {parameter for parameter, count in occurrence_count.items() if count > 1}
    initialized_parameters = {}

    def init_fn(module: nn.Module):
        for name, parameter in module.named_parameters(recurse=False):
            if parameter in shared_parameters:
                module._parameters[name] = initialized_parameters.setdefault(
                    parameter,
                    nn.Parameter(
                        torch.empty_like(parameter.data, device=device),
                        requires_grad=parameter.requires_grad,
                    ),
                )
            else:
                module._parameters[name] = nn.Parameter(
                    torch.empty_like(parameter.data, device=device),
                    requires_grad=parameter.requires_grad,
                )

    return init_fn


def get_fsdp_wrap_policy(model: PreTrainedModel, is_lora_model=False):
    transformer_layer_classes = set()
    for module_name in model._no_split_modules:
        module_class = get_module_class_from_name(model, module_name)
        if module_class is None:
            raise Exception(f"Cannot find {module_name} in pretrained model.")
        transformer_layer_classes.add(module_class)

    policies = []

    if is_lora_model:

        def lambda_policy_fn(module):
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        policies.append(partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn))

    policies.append(
        partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_classes,
        )
    )

    return partial(_or_policy, policies=policies)


@torch.no_grad()
def offload_fsdp_model(model: FSDP, empty_cache: bool = True):
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"

    for handle in model._all_handles:
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to("cpu", non_blocking=True)
        flat_param._local_shard = flat_param.data
        assert id(flat_param._local_shard) != id(flat_param.data)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_model(model: FSDP, empty_cache: bool = True):
    _lazy_init(model, model)
    assert model._is_root, "Only support root model loading to GPU"

    for handle in model._all_handles:
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        handle.flat_param_to("cuda", non_blocking=True)
        flat_param._local_shard = flat_param.data

    if empty_cache:
        gc.collect()


@torch.no_grad()
def offload_fsdp_optimizer(optimizer: Optimizer, empty_cache: bool = True):
    if not optimizer.state:
        return

    for parameter_group in optimizer.param_groups:
        for parameter in parameter_group["params"]:
            state = optimizer.state[parameter]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_optimizer(optimizer: Optimizer, empty_cache: bool = True):
    if not optimizer.state:
        return

    for parameter_group in optimizer.param_groups:
        for parameter in parameter_group["params"]:
            state = optimizer.state[parameter]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cuda", non_blocking=True)

    if empty_cache:
        gc.collect()


@torch.no_grad()
def offload_fsdp_submodule(module: FSDP, empty_cache: bool = True):
    for handle in _traversal_utils._get_fsdp_handles(module):
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to("cpu", non_blocking=True)
        flat_param._local_shard = flat_param.data

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_submodule(module: FSDP, empty_cache: bool = True):
    for handle in _traversal_utils._get_fsdp_handles(module):
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        handle.flat_param_to("cuda", non_blocking=True)
        flat_param._local_shard = flat_param.data

    if empty_cache:
        gc.collect()