import copy
import heapq
from itertools import chain
from typing import Optional, Tuple

import torch
from tensordict import TensorDict
from torch import distributed as dist

from ..protocol import DataProto


class Set:
    def __init__(self) -> None:
        self.sum = 0
        self.items = []

    def add(self, idx: int, val: int):
        self.items.append((idx, val))
        self.sum += val

    def merge(self, other):
        for idx, val in other.items:
            self.items.append((idx, val))
            self.sum += val

    def __lt__(self, other):
        if self.sum != other.sum:
            return self.sum < other.sum
        if len(self.items) != len(other.items):
            return len(self.items) < len(other.items)
        return self.items < other.items


class State:
    def __init__(self, items: list[Tuple[int, int]], k: int) -> None:
        self.k = k
        self.sets = [Set() for _ in range(k)]
        assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"

        for position, (index, length) in enumerate(items):
            self.sets[position].add(idx=index, val=length)

        self.sets = sorted(self.sets, reverse=True)

    def get_partitions(self):
        result = []
        for current_set in self.sets:
            result.append([index for index, _ in current_set.items])
        return result

    def merge(self, other):
        for position in range(self.k):
            self.sets[position].merge(other.sets[self.k - position - 1])
        self.sets = sorted(self.sets, reverse=True)

    @property
    def spread(self) -> int:
        return self.sets[0].sum - self.sets[-1].sum

    def __lt__(self, other):
        if self.spread != other.spread:
            return self.spread > other.spread
        return self.sets[0] > other.sets[0]

    def __repr__(self) -> str:
        chunks = []
        for current_set in self.sets:
            chunks.append("{" + ",".join(str(length) for _, length in current_set.items) + "}")
        return "[" + ",".join(chunks) + "]"


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    ordered = sorted((length, index) for index, length in enumerate(seqlen_list))
    pending = []

    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for start in range(0, len(ordered), k_partitions):
            group = []
            for offset in range(k_partitions):
                length, index = ordered[start + offset]
                group.append((index, length))
            heapq.heappush(pending, State(items=group, k=k_partitions))
    else:
        for length, index in ordered:
            heapq.heappush(pending, State(items=[(index, length)], k=k_partitions))

    while len(pending) > 1:
        left = heapq.heappop(pending)
        right = heapq.heappop(pending)
        left.merge(right)
        heapq.heappush(pending, left)

    partitions = pending[0].get_partitions()

    if equal_size:
        for partition in partitions:
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )

    return partitions


def greedy_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    offset = sum(seqlen_list) + 1 if equal_size else 0
    entries = [(length + offset, index) for index, length in enumerate(seqlen_list)]

    partitions = [[] for _ in range(k_partitions)]
    totals = [0 for _ in range(k_partitions)]

    for length, index in entries:
        target = None
        for partition_index in range(k_partitions):
            if target is None or totals[partition_index] < totals[target]:
                target = partition_index
        partitions[target].append(index)
        totals[target] += length

    if equal_size:
        for partition in partitions:
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )

    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool) -> list[list[int]]:
    assert len(seqlen_list) >= k_partitions, (
        f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"
    )

    def validate(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        all_indices = set()
        ordered_partitions = [None] * k_partitions

        for partition_index, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {partition_index}-th partition is empty"
            all_indices.update(partition)
            ordered_partitions[partition_index] = sorted(partition)

        assert all_indices == set(range(len(seqlen_list)))
        return ordered_partitions

    result = karmarkar_karp(
        seqlen_list=seqlen_list,
        k_partitions=k_partitions,
        equal_size=equal_size,
    )
    return validate(result)


def log_seqlen_unbalance(seqlen_list: list[int], partitions: list[list[int]], prefix: str) -> dict[str, float]:
    partition_count = len(partitions)
    batch_size = len(seqlen_list) // partition_count

    minimum = None
    maximum = None
    total = 0

    for start in range(0, len(seqlen_list), batch_size):
        current = sum(seqlen_list[start : start + batch_size])
        if minimum is None or current < minimum:
            minimum = current
        if maximum is None or current > maximum:
            maximum = current
        total += current

    balanced_totals = []
    for partition in partitions:
        balanced_totals.append(sum(seqlen_list[index] for index in partition))

    balanced_minimum = min(balanced_totals)
    balanced_maximum = max(balanced_totals)

    return {
        f"{prefix}/min": minimum,
        f"{prefix}/max": maximum,
        f"{prefix}/minmax_diff": maximum - minimum,
        f"{prefix}/balanced_min": balanced_minimum,
        f"{prefix}/balanced_max": balanced_maximum,
        f"{prefix}/mean": total / len(partitions),
    }


def ceildiv(a: float, b: float) -> int:
    return int((a + b - 1) // b)


def get_reverse_idx(idx: torch.Tensor) -> torch.Tensor:
    reverse_idx = torch.zeros_like(idx)
    reverse_idx[idx] = torch.arange(len(idx), device=idx.device)
    return reverse_idx


def rearrange_tensor(tensor: torch.Tensor, indices: list[list[int]]) -> torch.Tensor:
    flattened_indices = list(chain.from_iterable(indices))
    index_tensor = torch.tensor(flattened_indices, dtype=torch.long)
    return tensor[index_tensor]


def rearrange_tensordict(tensordict: TensorDict, indices: list[list[int]]) -> TensorDict:
    flattened_indices = list(chain.from_iterable(indices))
    index_tensor = torch.tensor(flattened_indices, dtype=torch.long)
    return tensordict[index_tensor]


def rearrange_dataproto(dataproto: DataProto, indices: list[list[int]]) -> DataProto:
    flattened_indices = list(chain.from_iterable(indices))
    reordered_non_tensor_batch = {}

    for key, value in dataproto.non_tensor_batch.items():
        reordered_non_tensor_batch[key] = value[flattened_indices]

    return DataProto(
        batch=rearrange_tensordict(dataproto.batch, indices),
        non_tensor_batch=reordered_non_tensor_batch,
        meta_info=copy.deepcopy(dataproto.meta_info),
    )


def restore_dataproto(dataproto: DataProto, indices: list[list[int]]) -> DataProto:
    flattened_indices = list(chain.from_iterable(indices))
    reverse_idx = get_reverse_idx(torch.tensor(flattened_indices, dtype=torch.long))
    restored_non_tensor_batch = {}

    for key, value in dataproto.non_tensor_batch.items():
        restored_non_tensor_batch[key] = value[reverse_idx.numpy()]

    return DataProto(
        batch=dataproto.batch[reverse_idx],
        non_tensor_batch=restored_non_tensor_batch,
        meta_info=copy.deepcopy(dataproto.meta_info),
    )