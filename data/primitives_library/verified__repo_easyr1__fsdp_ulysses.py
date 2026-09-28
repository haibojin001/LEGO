from torch.distributed.device_mesh import DeviceMesh

from ...protocol import DataProto, all_gather_data_proto
from ...utils.ulysses import (
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
)
from .base import BaseShardingManager


class FSDPUlyssesShardingManager(BaseShardingManager):
    """Coordinates FSDP partitioning with Ulysses sequence parallelism."""

    def __init__(self, device_mesh: DeviceMesh):
        super().__init__()
        self.device_mesh = device_mesh

    def __enter__(self):
        if self.device_mesh is not None:
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            sequence_parallel_mesh = self.device_mesh["sp"]
            set_ulysses_sequence_parallel_group(sequence_parallel_mesh.get_group())

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device_mesh is not None:
            set_ulysses_sequence_parallel_group(self.prev_sp_group)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """Gather sequence-parallel shards so all SP ranks receive the same data."""
        if self.device_mesh is not None:
            sequence_parallel_mesh = self.device_mesh["sp"]
            all_gather_data_proto(
                data,
                size=sequence_parallel_mesh.size(),
                group=sequence_parallel_mesh.get_group(),
            )
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Restore the partitioning expected by FSDP after sequence-parallel work."""
        if self.device_mesh is not None:
            sequence_parallel_mesh = self.device_mesh["sp"]
            data = data.chunk(chunks=sequence_parallel_mesh.size())[
                sequence_parallel_mesh.get_local_rank()
            ]
        return data