# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg398::random.getstate+torch.get_rng_state+torch.save
# name: random_torch_primitive
# summary: Uses random.getstate, torch.get_rng_state, torch.save across 2 repos
# anchor_symbols: ['random.getstate', 'torch.get_rng_state', 'torch.save']
# observed in 2 repos: ['freud14__poutyne', 'ruc-datalab__DeepAnalyze']...

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyrl-train/skyrl_train/distributed/strategy.py::DistributedStrategy.get_rng_state ---
def get_rng_state():
        """Get current RNG state for reproducibility"""
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        # Only save CUDA RNG state if CUDA is available and being used
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            rng_state["cuda"] = torch.cuda.get_rng_state()

        return rng_state

# --- from freud14__poutyne::poutyne/utils.py::save_random_states ---
def save_random_states(f: str | os.PathLike | BinaryIO | IO[bytes]):
    """
    Save Python, Numpy and Pytorch's (both CPU and GPU) random states.

    Args:
        f (str | os.PathLike | BinaryIO | IO[bytes]): a file-like object (has to implement write and flush) or
            a string or os.PathLike object containing a file name.
    """
    torch.save(
        {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        f,
        pickle_module=pickle,
    )

# --- from ruc-datalab__DeepAnalyze::deepanalyze/SkyRL/skyrl-train/skyrl_train/distributed/fsdp_strategy.py::FSDPStrategy.save_ckpt ---
def save_ckpt(
        self,
        model,
        ckpt_dir,
        global_step,
        node_local_rank,
        optimizer=None,
        scheduler=None,
        client_state={},
        tag=None,
        tokenizer=None,
    ):
        """Save model checkpoint for FSDP"""
        import warnings
        from torch.distributed.fsdp import (
            ShardedStateDictConfig,
            ShardedOptimStateDictConfig,
            StateDictType,
        )

        if node_local_rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)

        # Wait for checkpoint directory to be created.
        dist.barrier()

        # Extract the actual model for saving
        if isinstance(model, Actor):
            save_model = model.model
        else:
            save_model = model

        if self.fsdp_strategy not in ("fsdp", "fsdp2"):
            raise ValueError(f"Unsupported FSDP strategy: {self.fsdp_strategy}")

        # Set up state dict configurations for sharded saving
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

        # Define paths for saving individual rank files
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

        # Save using appropriate FSDP context
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(
                save_model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
            ):
                # Get and save model state dict
                model_state_dict = save_model.state_dict()
                self.print(
                    f"[rank-{rank}]: Saving model to {os.path.abspath(model_path)}"
                )
                torch.save(model_state_dict, model_path)

                # Get and save optimizer state dict if optimizer is provided
                optimizer_state_dict = {}
                if optimizer is not None:
                    optimizer_state_dict = optimizer.state_dict()
                self.print(
                    f"[rank-{rank}]: Saving optim to {os.path.abspath(optim_path)}"
                )
                torch.save(optimizer_state_dict, optim_path)

                # Get scheduler state dict if scheduler is provided
                lr_scheduler_state_dict = {}
                if scheduler is not None:
                    lr_scheduler_state_dict = scheduler.state_dict()

                # Create extra state dict with client state and any additional info
                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "client_state": client_state,
                    "tag": tag,
                    "fsdp_strategy": self.fsdp_strategy,
                    "world_size": world_size,
                    "rank": rank,
                    "global_step": global_step,
                    "rng": self.get_rng_state(),  # Add RNG state for reproducibility
                }

                # Save extra state
                self.print(
                    f"[rank-{rank}]: Saving extra_state to {os.path.abspath(extra_path)}"
                )
                torch.save(extra_state_dict, extra_path)

                # Garbage collect temporary buffers from materializing the state dicts
                gc.collect()

        if self.is_rank_0():
            config_save_model = self._unwrap_model(model)
            self.save_hf_configs(config_save_model, ckpt_dir, tokenizer)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(ckpt_dir, "fsdp_config.json")
            with open(fsdp_config_path, "w") as f:
                json.dump(
                    {
                        "fsdp_strategy": self.fsdp_strategy,
                        "world_size": self.world_size,
                    },
                    f,
                    indent=4,
                )

        # Final barrier to ensure all operations complete
        dist.barrier()
        torch.cuda.synchronize()
        self.print(f"[rank-{rank}]: Checkpoint saved to {ckpt_dir}")
