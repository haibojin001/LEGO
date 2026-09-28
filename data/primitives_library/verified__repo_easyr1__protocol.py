import copy
import io
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from tensordict import TensorDict
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader

try:
    import ray
except Exception:
    ray = None

try:
    from .utils.py_functional import union_two_dict
except Exception:
    def union_two_dict(dict1, dict2):
        output = dict(dict1)
        for key, value in dict2.items():
            if key in output and output[key] != value:
                raise ValueError(f"Key already exists: {key}.")
            output[key] = value
        return output

try:
    import tensordict

    tensordict.set_lazy_legacy(False).set()
except Exception:
    pass


__all__ = ["DataProto", "union_tensor_dict"]


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int) -> tuple["DataProto", int]:
    assert isinstance(data, DataProto), "data must be a DataProto"

    if len(data) % size_divisor == 0:
        return data, 0

    pad_size = size_divisor - len(data) % size_divisor
    remaining_pad = pad_size
    padding_protos = []

    while remaining_pad > 0:
        take_size = min(remaining_pad, len(data))
        padding_protos.append(data[:take_size])
        remaining_pad -= take_size

    return DataProto.concat([data] + padding_protos), pad_size


def unpad_dataproto(data: "DataProto", pad_size: int) -> "DataProto":
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    if tensor_dict1.batch_size != tensor_dict2.batch_size:
        raise ValueError(
            f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
        )

    for key in tensor_dict2.keys():
        if key in tensor_dict1 and not torch.equal(tensor_dict1[key], tensor_dict2[key]):
            raise ValueError(f"Key already exists: {key}.")
        tensor_dict1[key] = tensor_dict2[key]

    return tensor_dict1


def union_numpy_dict(tensor_dict1: dict[str, NDArray], tensor_dict2: dict[str, NDArray]) -> dict[str, NDArray]:
    for key, value in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict1[key], np.ndarray)
            assert isinstance(value, np.ndarray)
            if not np.all(tensor_dict1[key] == value):
                raise ValueError(f"Key already exists: {key}.")
        tensor_dict1[key] = value
    return tensor_dict1


def batch_collate(features: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if len(features) == 0:
        return {}

    output = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            output[key].append(value)
    return output


def fold_batch_dim(data: "DataProto", new_batch_size: int):
    batch_size = data.batch.batch_size[0]
    assert batch_size % new_batch_size == 0

    tensor = data.batch.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)

    non_tensor = data.non_tensor_batch
    for key, value in non_tensor.items():
        non_tensor[key] = np.reshape(value, (new_batch_size, -1, *value.shape[1:]))

    return DataProto(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


def collate_fn(data_items: list["DataProtoItem"]):
    batch = []
    non_tensor_batch = []

    for data in data_items:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)

    batch = torch.stack(batch).contiguous()
    non_tensor_batch = batch_collate(non_tensor_batch)
    non_tensor_batch = {
        key: np.array(value, dtype=object) for key, value in non_tensor_batch.items()
    }

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


@dataclass
class DataProtoItem:
    batch: Optional[TensorDict] = None
    non_tensor_batch: dict[str, NDArray] = field(default_factory=dict)
    meta_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProto:
    batch: Optional[TensorDict] = None
    non_tensor_batch: dict[str, NDArray] = field(default_factory=dict)
    meta_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.check_consistency()

    def __len__(self) -> int:
        if self.batch is not None:
            return self.batch.batch_size[0]
        if self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            key = next(iter(self.non_tensor_batch))
            return self.non_tensor_batch[key].shape[0]
        return 0

    def __getitem__(
        self, item: Union[int, slice, list[int], np.ndarray, torch.Tensor]
    ) -> Union["DataProto", DataProtoItem]:
        if isinstance(item, slice):
            return self.slice_select(item.start, item.stop, item.step)

        if isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.index_select(item)

        if isinstance(item, (int, np.integer)):
            batch = self.batch[item] if self.batch is not None else None
            non_tensor_batch = {
                key: value[item] for key, value in self.non_tensor_batch.items()
            }
            return DataProtoItem(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                meta_info=self.meta_info,
            )

        raise TypeError(f"Indexing with {type(item)} is not supported.")

    def __getstate__(self) -> tuple[bytes, dict[str, NDArray], dict[str, Any]]:
        batch = self.batch.contiguous().consolidate() if self.batch is not None else None
        buffer = io.BytesIO()
        torch.save(batch, buffer)
        return buffer.getvalue(), self.non_tensor_batch, self.meta_info

    def __setstate__(self, state: tuple[bytes, dict[str, NDArray], dict[str, Any]]) -> None:
        serialized_batch, non_tensor_batch, meta_info = state
        try:
            batch = torch.load(
                io.BytesIO(serialized_batch),
                weights_only=False,
                map_location="cpu",
            )
        except TypeError:
            batch = torch.load(io.BytesIO(serialized_batch), map_location="cpu")
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def save_to_disk(self, filepath: str) -> None:
        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_disk(filepath: str) -> "DataProto":
        with open(filepath, "rb") as file:
            return pickle.load(file)

    def print_size(self, prefix: str = "") -> None:
        tensor_size = 0
        if self.batch is not None:
            for value in self.batch.values():
                if isinstance(value, torch.Tensor):
                    tensor_size += value.element_size() * value.numel()

        numpy_size = 0
        for value in self.non_tensor_batch.values():
            numpy_size += value.nbytes

        tensor_size /= 1024**3
        numpy_size /= 1024**3
        message = (
            f"Size of tensordict: {tensor_size} GB, "
            f"size of non_tensor_batch: {numpy_size} GB."
        )
        print({prefix}, {message})

    def check_consistency(self):
        if self.batch is not None:
            assert isinstance(self.batch, TensorDict)
            if len(self.batch.batch_size) > 0:
                batch_size = self.batch.batch_size[0]
                for key, value in self.non_tensor_batch.items():
                    assert isinstance(value, np.ndarray), (
                        f"non_tensor_batch[{key}] must be a numpy array."
                    )
                    assert value.shape[0] == batch_size, (
                        f"Batch size mismatch for key {key}: "
                        f"{value.shape[0]} != {batch_size}."
                    )
        elif self.non_tensor_batch:
            batch_size = None
            for key, value in self.non_tensor_batch.items():
                assert isinstance(value, np.ndarray), (
                    f"non_tensor_batch[{key}] must be a numpy array."
                )
                if batch_size is None:
                    batch_size = value.shape[0]
                else:
                    assert value.shape[0] == batch_size, (
                        f"Batch size mismatch for key {key}: "
                        f"{value.shape[0]} != {batch_size}."
                    )
        return True

    @staticmethod
    def from_dict(
        tensors: Optional[dict[str, torch.Tensor]] = None,
        non_tensors: Optional[dict[str, NDArray]] = None,
        meta_info: Optional[dict[str, Any]] = None,
    ) -> "DataProto":
        tensors = {} if tensors is None else tensors
        non_tensors = {} if non_tensors is None else non_tensors
        meta_info = {} if meta_info is None else meta_info

        if tensors:
            first = next(iter(tensors.values()))
            batch_size = first.shape[:1]
            batch = TensorDict(tensors, batch_size=batch_size)
        else:
            batch = None

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensors,
            meta_info=meta_info,
        )

    @staticmethod
    def from_single_dict(
        data: dict[str, Any],
        meta_info: Optional[dict[str, Any]] = None,
    ) -> "DataProto":
        tensors = {}
        non_tensors = {}

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            elif isinstance(value, np.ndarray):
                non_tensors[key] = value
            else:
                raise ValueError(
                    f"Value of key {key} must be torch.Tensor or numpy.ndarray, "
                    f"but got {type(value)}."
                )

        return DataProto.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info=meta_info,
        )

    def to(self, device: Union[str, torch.device]) -> "DataProto":
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    def cpu(self) -> "DataProto":
        return self.to("cpu")

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> "DataProto":
        if device is None:
            return self.to("cuda")
        if isinstance(device, int):
            return self.to(f"cuda:{device}")
        return self.to(device)

    def clone(self) -> "DataProto":
        batch = self.batch.clone() if self.batch is not None else None
        non_tensor_batch = {
            key: value.copy() for key, value in self.non_tensor_batch.items()
        }
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=copy.deepcopy(self.meta_info),
        )

    def select(
        self,
        batch_keys: Optional[list[str]] = None,
        non_tensor_batch_keys: Optional[list[str]] = None,
        meta_info_keys: Optional[list[str]] = None,
    ) -> "DataProto":
        batch = self.batch
        if batch_keys is not None:
            batch = batch.select(*batch_keys) if batch is not None else None

        non_tensor_batch = self.non_tensor_batch
        if non_tensor_batch_keys is not None:
            non_tensor_batch = {
                key: self.non_tensor_batch[key] for key in non_tensor_batch_keys
            }

        meta_info = self.meta_info
        if meta_info_keys is not None:
            meta_info = {key: self.meta_info[key] for key in meta_info_keys}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    def pop(
        self,
        batch_keys: Optional[list[str]] = None,
        non_tensor_batch_keys: Optional[list[str]] = None,
        meta_info_keys: Optional[list[str]] = None,
    ) -> "DataProto":
        selected = self.select(batch_keys, non_tensor_batch_keys, meta_info_keys)

        if batch_keys is not None and self.batch is not None:
            self.batch = self.batch.exclude(*batch_keys)

        if non_tensor_batch_keys is not None:
            for key in non_tensor_batch_keys:
                self.non_tensor_batch.pop(key)

        if meta_info_keys is not None:
            for key in meta_info_keys:
                self.meta_info.pop(key)

        return selected

    def rename(self, old_keys: list[str], new_keys: list[str]) -> "DataProto":
        assert len(old_keys) == len(new_keys)
        if self.batch is not None:
            for old_key, new_key in zip(old_keys, new_keys):
                self.batch.rename_key_(old_key, new_key)
        return self

    def union(self, other: "DataProto") -> "DataProto":
        if self.batch is None:
            self.batch = other.batch
        elif other.batch is not None:
            self.batch = union_tensor_dict(self.batch, other.batch)

        self.non_tensor_batch = union_numpy_dict(
            self.non_tensor_batch,
            other.non_tensor_batch,
        )
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    @staticmethod
    def concat(data: list["DataProto"]) -> "DataProto":
        if len(data) == 0:
            return DataProto()

        batches = [item.batch for item in data]
        if all(batch is None for batch in batches):
            batch = None
        elif any(batch is None for batch in batches):
            raise ValueError("Cannot concatenate DataProto objects with mixed tensor batches.")
        else:
            batch = torch.cat(batches, dim=0)

        keys = data[0].non_tensor_batch.keys()
        for item in data[1:]:
            if item.non_tensor_batch.keys() != keys:
                raise ValueError("All DataProto objects must have identical non-tensor keys.")

        non_tensor_batch = {
            key: np.concatenate([item.non_tensor_batch[key] for item in data], axis=0)
            for key in keys
        }

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=data[0].meta_info,
        )

    def slice_select(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: Optional[int] = None,
    ) -> "DataProto":
        item = slice(start, end, step)
        batch = self.batch[item] if self.batch is not None else None
        non_tensor_batch = {
            key: value[item] for key, value in self.non_tensor_batch.items()
        }
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=self.meta_info,
        )

    def index_select(
        self,
        indices: Union[list[int], np.ndarray, torch.Tensor],
    ) -> "DataProto":
        tensor_indices = indices
        if isinstance(indices, np.ndarray):
            tensor_indices = torch.from_numpy(indices)
        elif isinstance(indices, list):
            tensor_indices = torch.tensor(indices, dtype=torch.long)

        batch = self.batch[tensor_indices] if self.batch is not None else None

        numpy_indices = indices
        if isinstance(indices, torch.Tensor):
            numpy_indices = indices.detach().cpu().numpy()

        non_tensor_batch = {
            key: value[numpy_indices] for key, value in self.non_tensor_batch.items()
        }
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=self.meta_info,
        )

    def split(self, split_size: int) -> list["DataProto"]:
        assert split_size > 0
        return [
            self.slice_select(start, min(start + split_size, len(self)))
            for start in range(0, len(self), split_size)
        ]

    def chunk(self, chunks: int) -> list["DataProto"]:
        assert chunks > 0
        if len(self) == 0:
            return []

        indices = np.array_split(np.arange(len(self)), chunks)
        return [self.index_select(index) for index in indices if len(index) > 0]

    def repeat(self, repeat_times: int, interleave: bool = True) -> "DataProto":
        assert repeat_times >= 0
        if repeat_times == 0:
            return self[:0]

        if interleave:
            indices = np.repeat(np.arange(len(self)), repeat_times)
        else:
            indices = np.tile(np.arange(len(self)), repeat_times)

        return self.index_select(indices)

    def reorder(self, indices: Union[list[int], np.ndarray, torch.Tensor]) -> "DataProto":
        selected = self.index_select(indices)
        self.batch = selected.batch
        self.non_tensor_batch = selected.non_tensor_batch
        return self

    def make_iterator(
        self,
        mini_batch_size: int,
        epochs: int,
        seed: Optional[int] = None,
        dataloader_kwargs: Optional[dict[str, Any]] = None,
    ):
        kwargs = {} if dataloader_kwargs is None else dict(dataloader_kwargs)
        kwargs.setdefault("batch_size", mini_batch_size)
        kwargs.setdefault("collate_fn", collate_fn)
        kwargs.setdefault("shuffle", True)

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            kwargs.setdefault("generator", generator)

        for _ in range(epochs):
            dataloader = DataLoader(self, **kwargs)
            for batch in dataloader:
                yield batch

    def allgather(
        self,
        process_group: Optional[ProcessGroup] = None,
        group_name: Optional[str] = None,
    ) -> "DataProto":
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return self

        world_size = dist.get_world_size(group=process_group)
        if world_size == 1:
            return self

        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, self, group=process_group)
        return DataProto.concat(gathered)