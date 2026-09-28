import math
import os
import platform
import warnings
from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

from ultralytics.utils.torch_utils import autocast as smart_amp_autocast
from ultralytics.utils.torch_utils import copy_attr, initialize_weights, is_parallel, model_info, scale_img, time_sync

from utils.general import LOGGER, check_version, colorstr, file_date, git_describe

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "-1"))
RANK = int(os.getenv("RANK", "-1"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

try:
    import thop
except ImportError:
    thop = None

warnings.filterwarnings("ignore", category=UserWarning)


def smart_inference_mode(torch_1_9=check_version(torch.__version__, "1.9.0")):
    """Return a decorator that applies inference mode when available, otherwise no-grad mode."""

    def decorator(function):
        context = torch.inference_mode if torch_1_9 else torch.no_grad
        return context()(function)

    return decorator


def smartCrossEntropyLoss(label_smoothing=0.0):
    """Construct cross entropy loss, using native label smoothing where supported."""
    if check_version(torch.__version__, "1.10.0"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        LOGGER.warning(f"label smoothing {label_smoothing} requires torch>=1.10.0")
    return nn.CrossEntropyLoss()


def smart_DDP(model):
    """Wrap a model in DistributedDataParallel with version-appropriate options."""
    assert not check_version(torch.__version__, "1.12.0", pinned=True), (
        "torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. "
        "Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395"
    )
    arguments = {"device_ids": [LOCAL_RANK], "output_device": LOCAL_RANK}
    if check_version(torch.__version__, "1.11.0"):
        arguments["static_graph"] = True
    return DDP(model, **arguments)


def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
    """Build an optimizer with independently managed decay, normalization, and bias parameter groups."""
    groups = ([], [], [])
    normalization_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

    for _, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name == "bias":
                groups[2].append(parameter)
            elif parameter_name == "weight" and isinstance(module, normalization_types):
                groups[1].append(parameter)
            else:
                groups[0].append(parameter)

    if name in ("Adam", "AdamW", "Adamax", "NAdam", "RAdam"):
        optimizer = getattr(torch.optim, name)(groups[2], lr=lr, betas=(momentum, 0.999))
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(groups[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(groups[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(
            f"Optimizer {name} not found in list of available optimizers "
            "[Adam, AdamW, Adamax, NAdam, RAdam, RMSProp, SGD, auto]"
        )

    optimizer.add_param_group({"params": groups[0], "weight_decay": decay})
    optimizer.add_param_group({"params": groups[1], "weight_decay": 0.0})
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f"{len(groups[1])} weight(decay=0.0), {len(groups[0])} weight(decay={decay}), {len(groups[2])} bias"
    )
    return optimizer


def smart_resume(ckpt, optimizer, ema=None, weights="yolov5s.pt", epochs=300, resume=True):
    """Restore optimizer and EMA state from a checkpoint and determine the resumed epoch range."""
    best_fitness = 0.0
    start_epoch = ckpt["epoch"] + 1

    if ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        best_fitness = ckpt["best_fitness"]

    if ema is not None and ckpt.get("ema"):
        ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
        ema.updates = ckpt["updates"]

    if resume:
        assert start_epoch > 0, (
            f"{weights} training to {epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        )
        LOGGER.info(f"Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs")

    if epochs < start_epoch:
        LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += ckpt["epoch"]

    return best_fitness, start_epoch, epochs


def reshape_classifier_output(model, n=1000):
    """Replace the terminal classifier layer of a model to produce n classes."""
    from models.common import Classify

    name, module = list((model.model if hasattr(model, "model") else model).named_children())[-1]

    if isinstance(module, Classify):
        if module.linear.out_features != n:
            module.linear = nn.Linear(module.linear.in_features, n)
    elif isinstance(module, nn.Linear):
        if module.out_features != n:
            setattr(model, name, nn.Linear(module.in_features, n))
    elif isinstance(module, nn.Sequential):
        module_types = [type(x) for x in module]
        if nn.Linear in module_types:
            index = len(module_types) - 1 - module_types[::-1].index(nn.Linear)
            if module[index].out_features != n:
                module[index] = nn.Linear(module[index].in_features, n)
        elif nn.Conv2d in module_types:
            index = len(module_types) - 1 - module_types[::-1].index(nn.Conv2d)
            if module[index].out_channels != n:
                module[index] = nn.Conv2d(
                    module[index].in_channels,
                    n,
                    module[index].kernel_size,
                    module[index].stride,
                    bias=module[index].bias is not None,
                )


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Make nonzero distributed ranks wait until rank zero has completed a protected operation."""
    if local_rank not in (-1, 0):
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def select_device(device="", batch_size=0, newline=True):
    """Choose CPU, CUDA, or MPS execution device and log the resulting environment."""
    message = (
        f"YOLOv5 🚀 {git_describe() or file_date(__file__)} "
        f"Python-{platform.python_version()} torch-{torch.__version__} "
    )
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")
    cpu = device == "cpu"
    mps = device == "mps"

    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(",", "")), (
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"
        )

    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(",") if device else "0"
        count = len(devices)

        if count > 1 and batch_size > 0:
            assert batch_size % count == 0, f"batch-size {batch_size} not multiple of GPU count {count}"

        indent = " " * (len(message) + 1)
        for index, requested_device in enumerate(devices):
            properties = torch.cuda.get_device_properties(index)
            prefix = "" if index == 0 else indent
            message += (
                f"{prefix}CUDA:{requested_device} ({properties.name}, "
                f"{properties.total_memory / (1 << 20):.0f}MiB)\n"
            )
        selected = "cuda:0"
    elif mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        message += "MPS\n"
        selected = "mps"
    else:
        message += "CPU\n"
        selected = "cpu"

    if not newline:
        message = message.rstrip()

    LOGGER.info(message)
    return torch.device(selected)


def profile(input, ops, n=10, device=None):
    """Measure model operation parameters, FLOPs, memory, forward time, and backward time."""
    results = []

    if not isinstance(device, torch.device):
        device = select_device(device)

    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )

    inputs = input if isinstance(input, list) else [input]
    operations = ops if isinstance(ops, list) else [ops]

    for x in inputs:
        x = x.to(device)
        x.requires_grad = True

        for operation in operations:
            operation = operation.to(device) if hasattr(operation, "to") else operation
            if hasattr(operation, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16:
                operation = operation.half()

            forward_time = 0
            backward_time = 0
            timing = [0, 0, 0]

            try:
                flops = thop.profile(operation, inputs=(x,), verbose=False)[0] / 1e9 * 2
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    timing[0] = time_sync()
                    y = operation(x)
                    timing[1] = time_sync()

                    try:
                        output_sum = sum(item.sum() for item in y) if isinstance(y, list) else y
                        output_sum.sum().backward()
                        timing[2] = time_sync()
                    except Exception:
                        timing[2] = float("nan")

                    forward_time += (timing[1] - timing[0]) * 1000 / n
                    backward_time += (timing[2] - timing[1]) * 1000 / n

                memory = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                input_shape, output_shape = (
                    tuple(value.shape) if isinstance(value, torch.Tensor) else "list" for value in (x, y)
                )
                parameters = sum(parameter.numel() for parameter in operation.parameters()) if isinstance(
                    operation, nn.Module
                ) else 0

                LOGGER.info(
                    f"{parameters:12}{flops:12.4g}{memory:>14.3f}{forward_time:14.4g}{backward_time:14.4g}"
                    f"{input_shape!s:>24s}{output_shape!s:>24s}"
                )
                results.append([parameters, flops, memory, forward_time, backward_time, input_shape, output_shape])
            except Exception as error:
                LOGGER.warning(error)
                results.append(None)

            torch.cuda.empty_cache()

    return results


def de_parallel(model):
    """Return the underlying model when wrapped by data parallelism."""
    return model.module if is_parallel(model) else model


def sparsity(model):
    """Return the global fraction of model parameters whose values are zero."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    zero_count = sum((parameter == 0).sum() for parameter in model.parameters())
    return zero_count / parameter_count


def prune(model, amount=0.3):
    """Apply L1 unstructured pruning to convolution weights and permanently remove pruning masks."""
    import torch.nn.utils.prune as torch_prune

    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            torch_prune.l1_unstructured(module, name="weight", amount=amount)
            torch_prune.remove(module, "weight")

    LOGGER.info(f"Model pruned to {sparsity(model):.3g} global sparsity")


def fuse_conv_and_bn(conv, bn):
    """Create a convolution equivalent to a convolution followed by batch normalization."""
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False).to(conv.weight.device)

    conv_weight = conv.weight.clone().view(conv.out_channels, -1)
    scale = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(scale, conv_weight).view(fused.weight.shape))

    conv_bias = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    bn_bias = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(scale, conv_bias.reshape(-1, 1)).reshape(-1) + bn_bias)

    return fused


class EarlyStopping:
    """Stop training when validation fitness has not improved for a configured number of epochs."""

    def __init__(self, patience=30):
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float("inf")
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.best_epoch = epoch

        delta = epoch - self.best_epoch
        self.possible_stop = delta >= self.patience - 1
        stop = delta >= self.patience

        if stop:
            LOGGER.info(
                f"Stopping training early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping."
            )

        return stop


class ModelEMA:
    """Maintain an exponential moving average copy of a model's floating-point state."""

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))

        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    def update(self, model):
        """Update EMA parameters and floating buffers using the supplied training model."""
        self.updates += 1
        decay = self.decay(self.updates)
        model_state = de_parallel(model).state_dict()

        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value *= decay
                value += (1 - decay) * model_state[key].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Copy selected non-parameter model attributes into the EMA model."""
        copy_attr(self.ema, model, include, exclude)


def strip_optimizer(f="best.pt", s=""):
    """Remove optimizer and training-only state from a checkpoint and save a compact inference checkpoint."""
    checkpoint = torch.load(f, map_location=torch.device("cpu"))

    if checkpoint.get("ema"):
        checkpoint["model"] = checkpoint["ema"]

    for key in ("optimizer", "best_fitness", "ema", "updates"):
        checkpoint[key] = None

    checkpoint["epoch"] = -1
    checkpoint["model"].half()

    for parameter in checkpoint["model"].parameters():
        parameter.requires_grad = False

    output = s or f
    torch.save(checkpoint, output)
    LOGGER.info(f"Optimizer stripped from {f}, {os.path.getsize(output) / 1e6:.1f}MB")