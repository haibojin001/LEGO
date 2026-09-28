# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg225::datetime.timedelta+torch.distributed.destroy_process_group+torch.distributed.init_process_group
# name: datetime_torch_primitive
# summary: Uses datetime.timedelta, torch.distributed.destroy_process_group, torch.distributed.init_process_group across 3 repos
# anchor_symbols: ['datetime.timedelta', 'torch.distributed.destroy_process_group', 'torch.distributed.init_process_group']
# observed in 3 repos: ['OML-Team__open-metric-learning', 'allenai__allennlp', 'run-house__kubetorch']...

# --- from OML-Team__open-metric-learning::tests/test_oml/test_ddp/utils.py::fn_ddp_wrapper ---
def fn_ddp_wrapper(
    rank: int, port: int, world_size: int, fn: Callable, *args: Tuple[Any, ...]  # type: ignore
) -> Any:  # type: ignore
    init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=120),
    )
    set_global_seed(1)
    torch.set_num_threads(1)
    res = fn(rank, world_size, *args)
    destroy_process_group()
    return res

# --- from allenai__allennlp::allennlp/common/testing/distributed_test.py::init_process ---
def init_process(
    process_rank: int,
    world_size: int,
    distributed_device_ids: List[int],
    func: Callable,
    func_args: Tuple = None,
    func_kwargs: Dict[str, Any] = None,
    primary_addr: str = "127.0.0.1",
    primary_port: int = 29500,
):
    assert world_size > 1

    global_rank = process_rank

    gpu_id = distributed_device_ids[process_rank]  # type: ignore

    if gpu_id >= 0:
        torch.cuda.set_device(int(gpu_id))
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{primary_addr}:{primary_port}",
            world_size=world_size,
            rank=global_rank,
        )
    else:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{primary_addr}:{primary_port}",
            world_size=world_size,
            rank=global_rank,
            timeout=datetime.timedelta(seconds=120),
        )

    func(global_rank, world_size, gpu_id, *(func_args or []), **(func_kwargs or {}))

    #  dist.barrier()
    dist.destroy_process_group()

# --- from run-house__kubetorch::python_client/kubetorch/data_store/pod_data_server.py::PodDataServer._init_nccl_process_group_global ---
def _init_nccl_process_group_global(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        timeout_seconds: int = KT_NCCL_TIMEOUT_SECONDS,
    ):
        """
        Initialize NCCL process group using global init_process_group.

        Uses TCPStore directly instead of env vars. Acquires semaphore to
        serialize NCCL operations (PyTorch only supports one global process
        group at a time).

        Yields the process group and cleans it up on exit.
        """
        from datetime import timedelta

        import torch.distributed as dist
        from torch.distributed import TCPStore

        process_group = None
        semaphore_acquired = False

        try:
            # Acquire semaphore to serialize NCCL operations
            semaphore_acquired = self._nccl_semaphore.acquire(timeout=self._nccl_semaphore_timeout)
            if not semaphore_acquired:
                raise RuntimeError(
                    f"Timeout waiting for NCCL semaphore after {self._nccl_semaphore_timeout}s. "
                    "Another NCCL operation may be stuck."
                )

            # Create explicit TCPStore - no env vars needed
            store = TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=world_size,
                is_master=(rank == 0),
                timeout=timedelta(seconds=timeout_seconds),
            )

            # Always destroy existing process group to ensure clean state.
            # NOTE: We can't use new_group() for cross-process broadcasts because
            # it's a collective that requires all ranks in the CURRENT world to
            # participate - but the getter is NOT in the source's world.
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception as e:
                    logger.warning(f"Failed to destroy existing process group: {e}")

            # Initialize with explicit store
            dist.init_process_group(
                backend="nccl",
                store=store,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=timeout_seconds),
            )
            process_group = dist.group.WORLD

            yield process_group

        finally:
            if process_group is not None:
                try:
                    dist.destroy_process_group()
                except Exception as e:
                    logger.warning(f"Failed to destroy process group: {e}")

            # Release semaphore
            if semaphore_acquired:
                self._nccl_semaphore.release()
