import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Union, cast

import torch

logger = logging.getLogger(__name__)

DeviceType = Literal[
    "cuda", "mps", "xpu", "vacc", "npu", "mlu", "musa", "gcu", "rocm", "cpu"
]


@dataclass
class DeviceSpec:
    name: str
    is_available: Callable[[], bool]
    env_name: Optional[str]
    get_gpu_info_fn: Callable[[], Dict]
    device_count_fn: Callable[[], int]
    empty_cache_fn: Optional[Callable] = None
    preferred_dtype: Optional[torch.dtype] = None
    hf_accelerate: bool = False
    pytorch_device_name: Optional[str] = None
    gpu_count_fn: Optional[Callable[[], int]] = None


def _count_with_visible_devices(env_name: str, device_count: int) -> int:
    value = os.getenv(env_name)
    if value is None:
        return device_count
    devices = value.split(",") if value else []
    return min(device_count, len(devices))


def _mps_empty_cache():
    try:
        torch.mps.empty_cache()
    except RuntimeError as exc:
        if "invalid low watermark ratio" not in str(exc):
            raise


def _musa_gpu_count() -> int:
    module = torch.get_device_module(torch.device(torch.musa.current_device()))
    count = module.device_count()
    value = os.getenv("MUSA_VISIBLE_DEVICES")
    if value is None:
        return count
    devices = value.split(",") if value else []
    return min(count, len(devices))


def _rocm_gpu_count() -> int:
    count = torch.cuda.device_count()
    value = os.getenv("HIP_VISIBLE_DEVICES")
    if value is None:
        value = os.getenv("CUDA_VISIBLE_DEVICES")
    if value is None:
        return count
    devices = value.split(",") if value else []
    return min(count, len(devices))


def is_metax_available() -> bool:
    try:
        from pymxsml import mxSmlGetDeviceCount  # noqa: F401
    except ImportError:
        return False
    return True


def is_rocm_available() -> bool:
    try:
        return torch.cuda.is_available() and torch.version.hip is not None
    except (ImportError, AttributeError):
        return False


def is_cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_mps_available() -> bool:
    try:
        return torch.backends.mps.is_available()
    except ImportError:
        return False


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


def is_mlu_available() -> bool:
    try:
        import torch_mlu  # noqa: F401

        return torch.mlu.is_available()
    except ImportError:
        return False


def is_vacc_available() -> bool:
    try:
        import torch_vacc  # noqa: F401

        return torch.vacc.is_available()
    except ImportError:
        return False


def is_musa_available() -> bool:
    try:
        import torch_musa  # noqa: F401
        import torchada  # noqa: F401

        return torch.musa.is_available()
    except ImportError:
        return False


def is_gcu_available() -> bool:
    try:
        import torch_gcu  # noqa: F401

        return torch.gcu.is_available()
    except ImportError:
        return False


def _get_info_by_pynvml(gpu_id: int) -> Dict[str, float]:
    import pynvml
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlDeviceGetUtilizationRates,
    )

    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    name = nvmlDeviceGetName(handle)
    try:
        memory = pynvml.nvmlDeviceGetMemoryInfo_v2(handle)
    except (pynvml.NVMLError, AttributeError):
        memory = nvmlDeviceGetMemoryInfo(handle)
    try:
        utilization = nvmlDeviceGetUtilizationRates(handle).gpu
    except pynvml.NVMLError:
        utilization = 0
    return {
        "name": name,
        "total": memory.total,
        "used": memory.used,
        "free": memory.free,
        "util": utilization,
    }


def _get_info_by_torch(index: int) -> Dict[str, Any]:
    return {
        "name": f"CUDA GPU {index}",
        "total": 1,
        "used": 0,
        "free": 1,
        "util": 0,
    }


def _get_metax_gpu_mem_info(gpu_id: int) -> Dict[str, Union[str, int]]:
    from pymxsml import mxSmlGetDeviceInfo, mxSmlGetMemoryInfo

    device_info = mxSmlGetDeviceInfo(gpu_id)
    memory_info = mxSmlGetMemoryInfo(gpu_id)
    total = memory_info.vramTotal * 1024
    used = memory_info.vramUse * 1024
    return {
        "name": device_info.deviceName,
        "total": total,
        "used": used,
        "free": total - used,
        "util": 0,
    }


def _get_rocm_gpu_mem_info(gpu_id: int) -> Dict[str, Union[str, int]]:
    return {
        "name": f"ROCm GPU {gpu_id}",
        "total": 1,
        "used": 0,
        "free": 1,
        "util": 0,
    }


def _get_gcu_mem_info(gpu_id: int) -> Dict[str, float]:
    from pyefml import efmlGetDevInfo, efmlGetDevMem

    device_info = efmlGetDevInfo(gpu_id)
    memory_info = efmlGetDevMem(gpu_id)
    return {
        "name": device_info.productName,
        "total": memory_info.vramTotal,
        "used": memory_info.vramUse,
        "free": memory_info.vramTotal - memory_info.vramUse,
        "util": 0,
    }


def _collect_gpu_info_with_env(
    env_name: str,
    device_count_fn: Callable[[], int],
    get_mem_info_fn: Callable[[int], Dict],
    init_fn: Optional[Callable] = None,
    shutdown_fn: Optional[Callable] = None,
) -> Dict:
    try:
        if init_fn is not None:
            init_fn()

        result: Dict[str, Dict] = {}
        count = device_count_fn()
        value = os.getenv(env_name)

        if not value:
            for index in range(count):
                result[f"gpu-{index}"] = get_mem_info_fn(index)
        else:
            for logical_index, device_id in enumerate(value.split(",")):
                try:
                    physical_index = int(device_id.strip())
                except ValueError:
                    continue
                if 0 <= physical_index < count:
                    result[f"gpu-{logical_index}"] = get_mem_info_fn(physical_index)

        return result
    except Exception as exc:
        logger.error(f"Fail to get GPU info: {exc}")
        return {}
    finally:
        if shutdown_fn is not None:
            try:
                shutdown_fn()
            except Exception:
                pass


def get_nvidia_gpu_info() -> Dict[str, Any]:
    if is_metax_available():
        try:
            from pymxsml import mxSmlGetDeviceCount

            return _collect_gpu_info_with_env(
                "CUDA_VISIBLE_DEVICES",
                mxSmlGetDeviceCount,
                _get_metax_gpu_mem_info,
            )
        except Exception as exc:
            logger.error(f"Fail to get GPU info: {exc}")
            return {}

    try:
        from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

        return _collect_gpu_info_with_env(
            "CUDA_VISIBLE_DEVICES",
            nvmlDeviceGetCount,
            _get_info_by_pynvml,
            nvmlInit,
            nvmlShutdown,
        )
    except Exception as exc:
        logger.error(f"Fail to get GPU info: {exc}")
        return {}


def get_rocm_gpu_info() -> Dict[str, Any]:
    return _collect_gpu_info_with_env(
        "HIP_VISIBLE_DEVICES",
        torch.cuda.device_count,
        _get_rocm_gpu_mem_info,
    )


def get_gcu_gpu_info() -> Dict[str, Any]:
    try:
        from pyefml import efmlInit, efmlShutdown

        return _collect_gpu_info_with_env(
            "TOPS_VISIBLE_DEVICES",
            torch.gcu.device_count,
            _get_gcu_mem_info,
            efmlInit,
            efmlShutdown,
        )
    except Exception as exc:
        logger.error(f"Fail to get GPU info: {exc}")
        return {}


def _empty_gpu_info() -> Dict:
    return {}


def _cuda_count() -> int:
    return _count_with_visible_devices("CUDA_VISIBLE_DEVICES", torch.cuda.device_count())


def _xpu_count() -> int:
    return _count_with_visible_devices("ZE_AFFINITY_MASK", torch.xpu.device_count())


def _npu_count() -> int:
    return _count_with_visible_devices(
        "ASCEND_VISIBLE_DEVICES", torch.npu.device_count()
    )


def _mlu_count() -> int:
    return _count_with_visible_devices("MLU_VISIBLE_DEVICES", torch.mlu.device_count())


def _vacc_count() -> int:
    return _count_with_visible_devices(
        "VACC_VISIBLE_DEVICES", torch.vacc.device_count()
    )


def _gcu_count() -> int:
    return _count_with_visible_devices("TOPS_VISIBLE_DEVICES", torch.gcu.device_count())


def _one_device() -> int:
    return 1


def _no_device() -> int:
    return 0


DEVICE_SPECS = [
    DeviceSpec(
        name="rocm",
        is_available=is_rocm_available,
        env_name="HIP_VISIBLE_DEVICES",
        get_gpu_info_fn=get_rocm_gpu_info,
        device_count_fn=_rocm_gpu_count,
        empty_cache_fn=torch.cuda.empty_cache,
        preferred_dtype=torch.float16,
        hf_accelerate=True,
        pytorch_device_name="cuda",
        gpu_count_fn=_rocm_gpu_count,
    ),
    DeviceSpec(
        name="cuda",
        is_available=is_cuda_available,
        env_name="CUDA_VISIBLE_DEVICES",
        get_gpu_info_fn=get_nvidia_gpu_info,
        device_count_fn=_cuda_count,
        empty_cache_fn=torch.cuda.empty_cache,
        preferred_dtype=torch.float16,
        hf_accelerate=True,
        pytorch_device_name="cuda",
        gpu_count_fn=_cuda_count,
    ),
    DeviceSpec(
        name="mps",
        is_available=is_mps_available,
        env_name=None,
        get_gpu_info_fn=_empty_gpu_info,
        device_count_fn=_one_device,
        empty_cache_fn=_mps_empty_cache,
        preferred_dtype=torch.float16,
        hf_accelerate=False,
        pytorch_device_name="mps",
    ),
    DeviceSpec(
        name="xpu",
        is_available=is_xpu_available,
        env_name="ZE_AFFINITY_MASK",
        get_gpu_info_fn=_empty_gpu_info,
        device_count_fn=_xpu_count,
        empty_cache_fn=lambda: torch.xpu.empty_cache(),
        preferred_dtype=torch.bfloat16,
        hf_accelerate=True,
        pytorch_device_name="xpu",
        gpu_count_fn=_xpu_count,
    ),
    DeviceSpec(
        name="vacc",
        is_available=is_vacc_available,
        env_name="VACC_VISIBLE_DEVICES",
        get_gpu_info_fn=_empty_gpu_info,
        device_count_fn=_vacc_count,
        empty_cache_fn=lambda: torch.vacc.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=False,
        pytorch_device_name="vacc",
        gpu_count_fn=_vacc_count,
    ),
    DeviceSpec(
        name="npu",
        is_available=is_npu_available,
        env_name="ASCEND_VISIBLE_DEVICES",
        get_gpu_info_fn=_empty_gpu_info,
        device_count_fn=_npu_count,
        empty_cache_fn=lambda: torch.npu.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=True,
        pytorch_device_name="npu",
        gpu_count_fn=_npu_count,
    ),
    DeviceSpec(
        name="mlu",
        is_available=is_mlu_available,
        env_name="MLU_VISIBLE_DEVICES",
        get_gpu_info_fn=_empty_gpu_info,
        device_count_fn=_mlu_count,
        empty_cache_fn=lambda: torch.mlu.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=False,
        pytorch_device_name="mlu",
        gpu_count_fn=_mlu_count,
    ),
    DeviceSpec(
        name="musa",
        is_available=is_musa_available,
        env_name="MUSA_VISIBLE_DEVICES",
        get_gpu_info_fn=_empty_gpu_info,
        device_count_fn=_musa_gpu_count,
        empty_cache_fn=lambda: torch.musa.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=False,
        pytorch_device_name="musa",
        gpu_count_fn=_musa_gpu_count,
    ),
    DeviceSpec(
        name="gcu",
        is_available=is_gcu_available,
        env_name="TOPS_VISIBLE_DEVICES",
        get_gpu_info_fn=get_gcu_gpu_info,
        device_count_fn=_gcu_count,
        empty_cache_fn=lambda: torch.gcu.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=False,
        pytorch_device_name="gcu",
        gpu_count_fn=_gcu_count,
    ),
    DeviceSpec(
        name="cpu",
        is_available=lambda: True,
        env_name=None,
        get_gpu_info_fn=_empty_gpu_info,
        device_count_fn=_one_device,
        preferred_dtype=torch.float32,
        hf_accelerate=False,
        pytorch_device_name="cpu",
        gpu_count_fn=_no_device,
    ),
]


def _get_device_spec(device: DeviceType) -> DeviceSpec:
    for spec in DEVICE_SPECS:
        if spec.name == device:
            return spec
    raise ValueError(f"Unsupported device: {device}")


def get_available_devices() -> list[DeviceType]:
    return [
        cast(DeviceType, spec.name)
        for spec in DEVICE_SPECS
        if spec.is_available()
    ]


def get_available_device() -> DeviceType:
    devices = get_available_devices()
    if not devices:
        raise RuntimeError("No available device found")
    return devices[0]


def get_device_count(device: DeviceType) -> int:
    return _get_device_spec(device).device_count_fn()


def get_gpu_count(device: DeviceType) -> int:
    spec = _get_device_spec(device)
    if spec.gpu_count_fn is not None:
        return spec.gpu_count_fn()
    return spec.device_count_fn()


def get_gpu_info(device: DeviceType) -> Dict:
    return _get_device_spec(device).get_gpu_info_fn()


def empty_cache(device: DeviceType):
    fn = _get_device_spec(device).empty_cache_fn
    if fn is not None:
        fn()


def get_preferred_dtype(device: DeviceType) -> Optional[torch.dtype]:
    return _get_device_spec(device).preferred_dtype


def is_hf_accelerate_device(device: DeviceType) -> bool:
    return _get_device_spec(device).hf_accelerate


def is_hf_accelerate_available(device: DeviceType) -> bool:
    return is_hf_accelerate_device(device)


def get_pytorch_device_name(device: DeviceType) -> Optional[str]:
    return _get_device_spec(device).pytorch_device_name