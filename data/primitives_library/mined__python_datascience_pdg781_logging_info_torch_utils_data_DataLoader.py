# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg781::logging.info+torch.utils.data.DataLoader
# name: logging_torch_primitive
# summary: Uses logging.info, torch.utils.data.DataLoader across 2 repos
# anchor_symbols: ['logging.info', 'torch.utils.data.DataLoader']
# observed in 2 repos: ['OML-Team__open-metric-learning', 'pykale__pykale']...

# --- from pykale__pykale::kale/loaddata/image_access.py::get_cifar ---
def get_cifar(cfg):
    """Gets training and validation data loaders for the CIFAR datasets

    Args:
        cfg (CfgNode): hyperparameters from configure file

    Examples:
        >>> train_loader, valid_loader = get_cifar(cfg)
    """
    logging.info("==> Preparing to load data " + cfg.DATASET.NAME + " at " + cfg.DATASET.ROOT)
    cifar_train_transform = get_transform("cifar", augment=True)
    cifar_test_transform = get_transform("cifar", augment=False)

    if cfg.DATASET.NAME == "CIFAR10":
        train_set = datasets.CIFAR10(
            cfg.DATASET.ROOT, train=True, download=cfg.DATASET.DOWNLOAD, transform=cifar_train_transform
        )
        valid_set = datasets.CIFAR10(
            cfg.DATASET.ROOT, train=False, download=cfg.DATASET.DOWNLOAD, transform=cifar_test_transform
        )
    elif cfg.DATASET.NAME == "CIFAR100":
        train_set = datasets.CIFAR100(
            cfg.DATASET.ROOT, train=True, download=cfg.DATASET.DOWNLOAD, transform=cifar_train_transform
        )
        valid_set = datasets.CIFAR100(
            cfg.DATASET.ROOT, train=False, download=cfg.DATASET.DOWNLOAD, transform=cifar_test_transform
        )
    else:
        raise NotImplementedError

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg.SOLVER.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, valid_loader

# --- from OML-Team__open-metric-learning::oml/ddp/patching.py::patch_dataloader_to_ddp ---
def patch_dataloader_to_ddp(loader: DataLoader) -> DataLoader:
    """
    Function inspects loader and modifies sampler for working in DDP mode.

    Note:
        We ALWAYS use the padding of samples (in terms of the number of batches or number of samples per epoch) in
        order to use the same amount of data for each device in DDP. Thus, the behavior with and without DDP may be
        slightly different (e.g. metrics values).

    """
    if is_ddp():
        kwargs_loader = extract_loader_parameters(loader, ignore_data_related_parameters=True)

        # If you don't spectify batch_sampler, PyTorch automatically creates default BatchSampler. In this case we
        # need convert to DDP only sampler (your custom sampler / default SequentialSampler or RandomSampler, which
        # PyTorch creates if sampler=None). We don't use `isinstance(...)` for `if` statement because we need exactly
        # class BatchSampler, ignoring any inheritance
        if type(loader.batch_sampler) is BatchSampler:
            ddp_sampler = DDPSamplerWrapper(
                sampler=loader.sampler, shuffle_samples_between_gpus=False, pad_data_to_num_gpus=True
            )
            patched_loader = DataLoader(
                dataset=loader.dataset,
                sampler=ddp_sampler,
                batch_size=loader.batch_size,
                drop_last=loader.drop_last,
                **kwargs_loader,
            )
            sampler_info = f"'{loader.sampler.__class__.__name__}' sampler"
        else:
            ddp_sampler = DDPSamplerWrapper(
                sampler=loader.batch_sampler, shuffle_samples_between_gpus=False, pad_data_to_num_gpus=True
            )
            patched_loader = DataLoader(dataset=loader.dataset, batch_sampler=ddp_sampler, **kwargs_loader)
            sampler_info = f"'{loader.batch_sampler.__class__.__name__}' batch sampler"

        logging.info(f"DataLoader with {sampler_info} is updated to DDP mode")
        return patched_loader
    else:
        warnings.warn(patch_dataloader_to_ddp.__name__, WarningDDP)
        return loader
