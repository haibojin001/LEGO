# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg399::random.setstate+torch.load+torch.set_rng_state
# name: random_torch_primitive
# summary: Uses random.setstate, torch.load, torch.set_rng_state across 2 repos
# anchor_symbols: ['random.setstate', 'torch.load', 'torch.set_rng_state']
# observed in 2 repos: ['freud14__poutyne', 'ruc-datalab__DeepAnalyze']...

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyrl-train/skyrl_train/distributed/strategy.py::DistributedStrategy.load_rng_state ---
def load_rng_state(rng_state):
        """Load RNG state for reproducibility"""
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        # Only restore CUDA RNG state if it was saved and CUDA is available
        if (
            "cuda" in rng_state
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 0
        ):
            torch.cuda.set_rng_state(rng_state["cuda"])

# --- from freud14__poutyne::poutyne/utils.py::load_random_states ---
def load_random_states(f: Any):
    """
    Load Python, Numpy and Pytorch's (both CPU and GPU) random states as saved by :func:`~poutyne.save_random_states()`.

    Args:
        f: a file-like object (has to implement :meth:`read`, :meth:`readline`, :meth:`tell`, and :meth:`seek`),
            or a string or os.PathLike object containing a file name
    """
    states = torch.load(f, pickle_module=pickle)
    torch.set_rng_state(states["cpu"])
    torch.cuda.set_rng_state_all(states["cuda"])
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyrl-train/skyrl_train/distributed/fsdp_strategy.py::FSDPStrategy.load_ckpt ---
def load_ckpt(
        self,
        model,
        ckpt_dir,
        optimizer=None,
        scheduler=None,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        """Load model checkpoint for FSDP"""
        import warnings
        from torch.distributed.fsdp import (
            ShardedStateDictConfig,
            ShardedOptimStateDictConfig,
            StateDictType,
        )

        if ckpt_dir is None:
            raise ValueError("ckpt_dir cannot be None")
        elif not os.path.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # Extract the actual model for loading
        load_model = model
        if isinstance(model, Actor):
            load_model = model.model

        # Define paths for loading individual rank files
        rank = self.get_rank()
        world_size = self.world_size
        model_path = os.path.join(
            ckpt_dir, f"model_world_size_{world_size}_rank_{rank}.pt"
        )
        optim_path = os.path.join(
            ckpt_dir, f"optim_world_size_{world_size}_rank_{rank}.pt"
        )
        extra_path = os.path.join(
            ckpt_dir, f"extra_state_world_size_{world_size}_rank_{rank}.pt"
        )

        # Check if checkpoint files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not os.path.exists(extra_path):
            raise FileNotFoundError(f"Extra state checkpoint not found: {extra_path}")

        # Optimizer path is optional since we may not save optimizer states initially
        optim_exists = os.path.exists(optim_path)

        self.print(f"[rank-{rank}]: Loading model from {os.path.abspath(model_path)}")
        self.print(
            f"[rank-{rank}]: Loading extra_state from {os.path.abspath(extra_path)}"
        )
        if optim_exists:
            self.print(
                f"[rank-{rank}]: Loading optim from {os.path.abspath(optim_path)}"
            )

        # Load state dictionaries from disk
        model_state_dict = torch.load(
            model_path, map_location="cpu", weights_only=False
        )
        extra_state_dict = torch.load(
            extra_path, map_location="cpu", weights_only=False
        )

        optimizer_state_dict = {}
        if optim_exists and load_optimizer_states and not load_module_only:
            optimizer_state_dict = torch.load(
                optim_path, map_location="cpu", weights_only=False
            )

        # Extract scheduler state from extra state
        lr_scheduler_state_dict = extra_state_dict.get("lr_scheduler", {})

        # Set up state dict configurations for sharded loading
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

        # Load using appropriate FSDP context
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(
                load_model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
            ):
                # Load model state dict
                load_model.load_state_dict(model_state_dict, strict=load_module_strict)
                self.print(f"[rank-{rank}]: Successfully loaded model state dict")

                # Load optimizer state dict if optimizer object is provided and loading is requested
                if (
                    optimizer is not None
                    and load_optimizer_states
                    and not load_module_only
                    and optimizer_state_dict
                ):
                    optimizer.load_state_dict(optimizer_state_dict)
                    self.print(f"[rank-{rank}]: Successfully loaded optimizer state")

                # Load scheduler state dict if scheduler object is provided and loading is requested
                if (
                    scheduler is not None
                    and load_lr_scheduler_states
                    and not load_module_only
                ):
                    scheduler.load_state_dict(lr_scheduler_state_dict)
                    self.print(f"[rank-{rank}]: Successfully loaded scheduler state")

        # Load RNG state for reproducibility
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

        # Wait for all ranks to finish loading
        dist.barrier()

        # Create states dict with extra information
        client_state = extra_state_dict.get("client_state", {})
        states = {
            "client_state": client_state,
            "tag": extra_state_dict.get("tag", tag),
            "fsdp_strategy": extra_state_dict.get("fsdp_strategy", self.fsdp_strategy),
            "world_size": extra_state_dict.get("world_size", world_size),
            "rank": extra_state_dict.get("rank", rank),
            "global_step": extra_state_dict.get(
                "global_step", 0
            ),  # Include global_step in return
        }

        self.print(f"[rank-{rank}]: Checkpoint loaded successfully from {ckpt_dir}")

        return ckpt_dir, states
