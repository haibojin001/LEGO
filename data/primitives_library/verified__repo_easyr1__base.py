import os
import random
import re
import string
import time
from typing import Any, Optional
from unittest.mock import patch

import ray
from ray.actor import ActorHandle
from ray.experimental.state.api import get_actor
from ray.util import list_named_actors
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from ..base import ClassWithInitArgs, ResourcePool, Worker, WorkerGroup
from ..base.decorator import MAGIC_ATTR

__all__ = ["Worker"]


def get_random_string(length: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    def generated(*args, **kwargs):
        dispatched_args, dispatched_kwargs = dispatch_fn(self, *args, **kwargs)
        result = execute_fn(method_name, *dispatched_args, **dispatched_kwargs)
        if blocking:
            result = ray.get(result)
        return collect_fn(self, result)

    return generated


def sort_placement_group_by_node_ip(pgs: list[PlacementGroup]) -> list[PlacementGroup]:
    node_addresses = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    addresses = {}

    for pg in pgs:
        state = ray._private.state.state.placement_group_table(pg.id)
        bundle_nodes = state["bundles_to_node_id"]
        node_id = bundle_nodes[0]
        addresses[pg.id] = node_addresses[node_id]

    return sorted(pgs, key=lambda item: addresses[item.id])


class RayResourcePool(ResourcePool):
    def __init__(
        self,
        process_on_nodes: list[int] = None,
        use_gpu: bool = True,
        name_prefix: str = "",
        max_colocate_count: int = 5,
        detached: bool = False,
    ) -> None:
        super().__init__(process_on_nodes, max_colocate_count)
        self.use_gpu = use_gpu
        self.name_prefix = name_prefix
        self.pgs = None
        self.detached = detached

    def get_placement_groups(
        self, strategy: str = "STRICT_PACK", name: Optional[str] = None
    ) -> list[PlacementGroup]:
        if self.pgs is not None:
            return self.pgs

        prefix = name or (
            f"{self.name_prefix}verl_group_{'_'.join(str(count) for count in self._store)}:"
        )
        bundles_per_group = []

        for process_count in self._store:
            if self.use_gpu:
                bundles = [
                    {"CPU": self.max_colocate_count, "GPU": 1}
                    for _ in range(process_count)
                ]
            else:
                bundles = [{"CPU": self.max_colocate_count} for _ in range(process_count)]
            bundles_per_group.append(bundles)

        lifetime = "detached" if self.detached else None
        groups = [
            placement_group(
                bundles=bundles,
                strategy=strategy,
                name=f"{prefix}{index}",
                lifetime=lifetime,
            )
            for index, bundles in enumerate(bundles_per_group)
        ]

        ray.get([group.ready() for group in groups])
        self.pgs = groups
        return groups


def extract_pg_from_exist(
    resource_pools: dict[str, RayResourcePool],
    src_role_names: list[str],
    resource_pool: RayResourcePool,
) -> list[PlacementGroup]:
    available = [
        pg
        for role_name, pool in resource_pools.items()
        if role_name in src_role_names
        for pg in pool.get_placement_groups()
    ]

    available.sort(key=lambda pg: pg.bundle_count, reverse=True)
    requested = sorted(
        ((count, index) for index, count in enumerate(resource_pool.store)),
        reverse=True,
    )

    selected = []
    cursor = 0
    for requested_count, original_index in requested:
        assert cursor < len(available), (
            f"no enough nodes for request: searching {cursor} th node"
        )
        candidate = available[cursor]
        assert requested_count <= candidate.bundle_count, (
            f"requesting {requested_count} processes, bundle count cannot satisfy"
        )
        selected.append((original_index, candidate))
        cursor += 1

    return [pg for _, pg in sorted(selected)]


def merge_resource_pool(rp1: RayResourcePool, rp2: RayResourcePool) -> RayResourcePool:
    assert rp1.use_gpu == rp2.use_gpu, (
        "Both RayResourcePool must either use_gpu or not"
    )
    assert rp1.max_colocate_count == rp2.max_colocate_count, (
        "Both RayResourcePool must has the same max_colocate_count"
    )
    assert rp1.n_gpus_per_node == rp2.n_gpus_per_node, (
        "Both RayResourcePool must has the same n_gpus_per_node"
    )
    assert rp1.detached == rp2.detached, (
        "Detached ResourcePool cannot be merged with non-detached ResourcePool"
    )

    merged = RayResourcePool(
        rp1.store + rp2.store,
        use_gpu=rp1.use_gpu,
        name_prefix=f"{rp1.name_prefix}_{rp2.name_prefix}",
        max_colocate_count=rp1.max_colocate_count,
        detached=rp1.detached,
    )
    merged.n_gpus_per_node = rp1.n_gpus_per_node
    merged.pgs = rp1.get_placement_groups() + rp2.get_placement_groups()
    return merged


class RayClassWithInitArgs(ClassWithInitArgs):
    def __init__(self, cls, *args, **kwargs) -> None:
        super().__init__(cls, *args, **kwargs)
        self._options = {}
        self._additional_resource = {}

    def set_additional_resource(self, additional_resource):
        self._additional_resource = additional_resource

    def update_options(self, options: dict):
        self._options.update(options)

    def __call__(
        self,
        placement_group: PlacementGroup,
        placement_group_bundle_idx: int,
        use_gpu: bool = True,
        num_gpus: int = 1,
        sharing_with: Worker = None,
    ) -> Any:
        if sharing_with is not None:
            node_id = ray.get(sharing_with.get_node_id.remote())
            cuda_visible_devices = ray.get(
                sharing_with.get_cuda_visible_devices.remote()
            )
            actor_options = {
                "scheduling_strategy": NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                )
            }
            return self.cls.options(**actor_options).remote(
                *self.args,
                cuda_visible_devices=cuda_visible_devices,
                **self.kwargs,
            )

        actor_options = {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=placement_group_bundle_idx,
            )
        }
        actor_options.update(self._options)

        if use_gpu:
            actor_options["num_gpus"] = num_gpus

        if len(self._additional_resource) > 1:
            actor_options.update(self._additional_resource)

        return self.cls.options(**actor_options).remote(*self.args, **self.kwargs)


class RayWorkerGroup(WorkerGroup):
    def __init__(
        self,
        resource_pool: RayResourcePool = None,
        ray_cls_with_init: RayClassWithInitArgs = None,
        bin_pack: bool = True,
        name_prefix: str = None,
        detached: bool = False,
        worker_names: list[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.ray_cls_with_init = ray_cls_with_init
        self.name_prefix = get_random_string(6) if name_prefix is None else name_prefix

        if worker_names is not None:
            assert self._is_init_with_detached_workers
            self._worker_names = worker_names

        if self._is_init_with_detached_workers:
            self._init_with_detached_workers(worker_names=worker_names)
        else:
            self._init_with_resource_pool(
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                bin_pack=bin_pack,
                detached=detached,
            )

        if ray_cls_with_init is not None:
            self._bind_work()

    @property
    def workers(self):
        return self._workers

    @property
    def worker_names(self):
        return self._worker_names

    def _init_with_resource_pool(
        self,
        resource_pool: RayResourcePool,
        ray_cls_with_init: RayClassWithInitArgs,
        bin_pack: bool = True,
        detached: bool = False,
    ) -> None:
        placement_groups = resource_pool.get_placement_groups()
        try:
            placement_groups = sort_placement_group_by_node_ip(placement_groups)
        except Exception:
            pass

        self._workers = []
        self._worker_names = []
        worker_index = 0

        for node_index, local_world_size in enumerate(resource_pool.store):
            pg = placement_groups[node_index]

            for local_rank in range(local_world_size):
                worker_name = f"{self.name_prefix}_worker_{worker_index}"

                if detached:
                    ray_cls_with_init.update_options(
                        {"name": worker_name, "lifetime": "detached"}
                    )

                bundle_index = local_rank
                if not bin_pack:
                    bundle_index = local_rank

                actor = ray_cls_with_init(
                    placement_group=pg,
                    placement_group_bundle_idx=bundle_index,
                    use_gpu=resource_pool.use_gpu,
                )
                self._workers.append(actor)
                self._worker_names.append(worker_name)
                worker_index += 1

    def _init_with_detached_workers(self, worker_names: list[str] = None) -> None:
        if worker_names is None:
            named = list_named_actors(all_namespaces=True)
            prefix = re.compile(rf"^{re.escape(self.name_prefix)}_worker_(\d+)$")
            matched = []

            for entry in named:
                if isinstance(entry, str):
                    name = entry
                    namespace = None
                else:
                    name = entry.get("name")
                    namespace = entry.get("namespace")

                if name is None:
                    continue

                match = prefix.match(name)
                if match is not None:
                    matched.append((int(match.group(1)), name, namespace))

            matched.sort(key=lambda item: item[0])
            worker_names = [item[1] for item in matched]
            namespaces = [item[2] for item in matched]
        else:
            namespaces = [None] * len(worker_names)

        self._workers = []
        self._worker_names = list(worker_names)

        for name, namespace in zip(worker_names, namespaces):
            handle = None
            last_error = None
            for _ in range(30):
                try:
                    if namespace is None:
                        handle = ray.get_actor(name)
                    else:
                        handle = ray.get_actor(name, namespace=namespace)
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(1)

            if handle is None:
                raise last_error
            self._workers.append(handle)

    def _is_worker_alive(self, worker):
        try:
            actor_id = worker._actor_id.hex()
            state = get_actor(actor_id)

            if state is None:
                return False

            if isinstance(state, dict):
                return state.get("state") == "ALIVE"

            return getattr(state, "state", None) == "ALIVE"
        except Exception:
            return False

    def _execute_all(self, method_name, *args, **kwargs):
        return [
            getattr(worker, method_name).remote(*args, **kwargs)
            for worker in self._workers
        ]

    def _execute_rank_zero(self, method_name, *args, **kwargs):
        if not self._workers:
            return None
        return getattr(self._workers[0], method_name).remote(*args, **kwargs)

    def _execute_one(self, method_name, *args, **kwargs):
        return self._execute_rank_zero(method_name, *args, **kwargs)

    def _bind_work(self):
        cls = self.ray_cls_with_init.cls

        for method_name in dir(cls):
            method = getattr(cls, method_name)

            if not hasattr(method, MAGIC_ATTR):
                continue

            metadata = getattr(method, MAGIC_ATTR)
            if not isinstance(metadata, dict):
                continue

            dispatch_fn = metadata.get("dispatch_fn")
            collect_fn = metadata.get("collect_fn")
            execute_fn = metadata.get("execute_fn")
            blocking = metadata.get("blocking", True)

            if dispatch_fn is None or collect_fn is None or execute_fn is None:
                continue

            setattr(
                self,
                method_name,
                func_generator(
                    self,
                    method_name,
                    dispatch_fn,
                    collect_fn,
                    execute_fn,
                    blocking,
                ),
            )

    def get_master_addr_port(self):
        if self._master_addr is None or self._master_port is None:
            if not self._workers:
                raise RuntimeError("Cannot infer master address without workers")

            master = self._workers[0]
            self._master_addr = ray.get(master.get_node_ip.remote())
            self._master_port = ray.get(master.get_free_port.remote())

        return self._master_addr, self._master_port