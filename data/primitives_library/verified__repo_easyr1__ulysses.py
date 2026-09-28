from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


_ULYSSES_SEQUENCE_PARALLEL_GROUP = None


def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup):
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:
    process_group = get_ulysses_sequence_parallel_group() if group is None else group
    if process_group:
        return dist.get_world_size(process_group)
    return 1


def get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int:
    process_group = get_ulysses_sequence_parallel_group() if group is None else group
    if process_group:
        return dist.get_rank(process_group)
    return 0


def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    padding_shape = list(x.shape)
    padding_shape[dim] = padding_size
    padding_tensor = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, padding_tensor], dim=dim)


def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    selection = [slice(None)] * len(x.shape)
    selection[dim] = slice(0, -padding_size)
    return x[selection]


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup = None,
) -> Tensor:
    process_group = get_ulysses_sequence_parallel_group() if group is None else group
    if not process_group:
        return x

    world_size = get_ulysses_sequence_parallel_world_size(process_group)
    output = SeqAllToAll.apply(process_group, x, head_dim, seq_dim)

    if unpadded_dim_size and unpadded_dim_size % world_size != 0:
        output = _unpad_tensor(output, seq_dim, output.size(seq_dim) - unpadded_dim_size)

    return output


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int, group: ProcessGroup = None) -> Tensor:
    process_group = get_ulysses_sequence_parallel_group() if group is None else group
    if not process_group:
        return x

    world_size = get_ulysses_sequence_parallel_world_size(process_group)
    seq_size = x.size(seq_dim)

    if seq_size % world_size != 0:
        x = _pad_tensor(x, seq_dim, world_size - seq_size % world_size)

    return SeqAllToAll.apply(process_group, x, seq_dim, head_dim, False)


def slice_input_tensor(x: Tensor, dim: int, padding: bool = True, group: ProcessGroup = None) -> Tensor:
    process_group = get_ulysses_sequence_parallel_group() if group is None else group
    world_size = dist.get_world_size(process_group)
    rank = get_ulysses_sequence_parallel_rank()
    dim_size = x.size(dim)

    if padding and dim_size % world_size:
        x = _pad_tensor(x, dim, world_size - dim_size % world_size)

    part_size = x.size(dim) // world_size
    slices = [slice(None)] * len(x.shape)
    slices[dim] = slice(rank * part_size, (rank + 1) * part_size)
    return x[slices].contiguous()


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    process_group = get_ulysses_sequence_parallel_group() if group is None else group
    world_size = dist.get_world_size(process_group)

    inputs = [part.contiguous() for part in torch.tensor_split(local_input, world_size, scatter_dim)]
    outputs = [torch.empty_like(inputs[0]) for _ in range(world_size)]
    handle = dist.all_to_all(outputs, inputs, group=process_group, async_op=async_op)

    if async_op:

        def wait():
            handle.wait()
            return torch.cat(outputs, dim=gather_dim).contiguous()

        return wait

    return torch.cat(outputs, dim=gather_dim).contiguous()


def all_gather_tensor(local_tensor: Tensor, group: Optional[dist.ProcessGroup] = None, async_op: bool = False):
    process_group = get_ulysses_sequence_parallel_group() if group is None else group
    world_size = dist.get_world_size(group=process_group)

    shape = list(local_tensor.shape)
    shape[0] *= world_size
    output = torch.empty(shape, dtype=local_tensor.dtype, device=local_tensor.device)
    dist.all_gather_into_tensor(output, local_tensor, group=process_group, async_op=async_op)
    return output


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        if ctx.async_op:
            gradient = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous()
        else:
            gradient = grad_output[0]

        return (
            None,
            all_to_all_tensor(gradient, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            None,
            None,
            None,
            None,
        )


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        ctx.sp_world_size = dist.get_world_size(group=group)
        ctx.sp_rank = dist.get_rank(group=group)

        local_shape = list(local_tensor.size())
        batch_size = local_shape[0]
        ctx.part_size = local_shape[gather_dim]

        gathered = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(gathered.split(batch_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size

        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),
            None,
            None,
            None,
            None,
        )


def gather_outputs_and_unpad(
    x: Tensor,
    gather_dim: int,
    unpad_dim: int = None,
    padding_size: int = 0,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
):
    process_group = get_ulysses_sequence_parallel_group() if group is None else group
    if not process_group:
        return x

    output = Gather.apply(process_group, x, gather_dim, grad_scaler)

    if unpad_dim is not None and padding_size:
        output = _unpad_tensor(output, unpad_dim, padding_size)

    return output