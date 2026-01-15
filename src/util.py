import logging
import os
from datetime import timedelta
from enum import auto, Enum, IntEnum

from omegaconf import DictConfig


class APIActions(Enum):
    RESPOND_TO_QUERIES = "respond_to_queries"
    GET_ENCODING = "get_encoding"
    GET_DECODING = "get_decoding"
    GET_TENSOR_ENCODING = "get_tensor_encoding"
    TRAIN_AGENT = "train_agent"
    GET_RANDOM_VALIDATION_EXAMPLE = "get_random_validation_example"
    RUN_EVALUATION = "run_evaluation"
    PROFILE_AGENT_TRAINING = "profile_agent_training"


class AgentAction(IntEnum):
    # Always keep this as first in enum.
    NO_ACTION_START = auto()
    GET_RESPONSES_FOR_QUERIES = auto()
    GET_ENCODINGS_FOR_QUERIES = auto()
    GET_DECODINGS_FOR_QUERIES = auto()
    GET_ENCODED_TENSORS_FOR_QUERIES = auto()
    TRAIN_AGENT = auto()
    RUN_EVALUATION = auto()
    PROFILE_AGENT_TRAINING = auto()
    # Always keep this as last in enum.
    NO_ACTION_END = auto()


class EvalBenchmark(Enum):
    """Supported evaluation benchmarks."""

    HUMANEVAL = "humaneval"


class DatasetType(Enum):
    OPENWEBTEXT = "openwebtext"
    AG_NEWS = "ag_news"
    WIKITEXT = "wikitext"
    FINANCEQA = "financeqa"
    MMLU_REDUX = "mmlu_redux"
    STACK_V2_PYTHON = "stack_v2_python"


def is_device_cpu(agent_device: str):
    return agent_device == "cpu"


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def is_rank_0():
    import torch.distributed as dist

    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_tensor_dtype_from_config(config: DictConfig):
    """Get torch dtype from config - imports torch only when called"""
    from torch import (
        bfloat16 as torch_bfloat16,
        float16 as torch_float16,
        float32 as torch_float32,
        float64 as torch_float64,
        int32 as torch_int32,
        int64 as torch_int64,
    )

    config_tensor_dtype: str = config.env.tensor_dtype
    if config_tensor_dtype == "int64":
        return torch_int64
    elif config_tensor_dtype == "int32":
        return torch_int32
    elif config_tensor_dtype == "float64":
        return torch_float64
    elif config_tensor_dtype == "float32":
        return torch_float32
    elif config_tensor_dtype == "float16":
        return torch_float16
    elif config_tensor_dtype == "bfloat16":
        return torch_bfloat16
    assert 0, f"Unsupported tensor dtype: {config_tensor_dtype}."
    return None


def get_logging_level_from_config(config: DictConfig) -> int:
    logging_level = config.env.logging_level
    if isinstance(logging_level, str):
        logging_level = logging.getLevelName(logging_level)
    return logging_level


def setup_torch_dist_init():
    """Initialize torch distributed across CUDA, ROCm, TPU, MPS, CPU."""
    import os
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    if not dist.is_available() or dist.is_initialized():
        return

    # --------------------
    # TPU (PJRT / torch_xla)
    # --------------------
    if os.environ.get("PJRT_DEVICE") == "TPU":
        import torch_xla.core.xla_model as xm
        # TPU runtime is already initialized by launcher
        device = xm.xla_device()
        print(
            f"[setup_torch_dist_init] TPU distributed ready "
            f"(rank={xm.get_ordinal()}, world_size={xm.world_size()})"
        )
        return

    # --------------------
    # CUDA / ROCm
    # --------------------
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=3600),
        )
        device = torch.device(f"cuda:{local_rank}")
        dist.barrier()
        print(
            f"[setup_torch_dist_init] CUDA/ROCm distributed initialized "
            f"(rank={dist.get_rank()}, world_size={dist.get_world_size()})"
        )
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        print("[setup_torch_dist_init] Single-process execution.")
        return
    # --------------------
    # CPU / Apple MPS
    # --------------------
    dist.init_process_group(
        backend="gloo",
        timeout=timedelta(seconds=3600),
    )
    device = torch.device("cpu")
    dist.barrier()
    print(
        f"[setup_torch_dist_init] Gloo distributed initialized "
        f"(rank={dist.get_rank()}, world_size={dist.get_world_size()})"
    )


def dist_barrier(device: "torch.device") -> None:
    """Helper method to synchronize agent if model is distributed."""
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return
    from src.logger import get_logger
    logger = get_logger()
    rank = dist.get_rank()
    logger.debug(
        f"Starting dist.barrier(), rank: {rank}, device: {device}",
        all_ranks=True,
    )
    if device.type == "cuda":
        dist.barrier(device_ids=([device.index] if device.type == "cuda" else None))
    else:
        dist.barrier()
    logger.debug(
        f"Done dist.barrier(), rank: {rank}, device: {device}",
        all_ranks=True,
    )


def get_device_info() -> dict:
    """Detect available device and backend type."""
    import torch

    device_info = {
        "device": "cpu",
        "backend": "cpu",
        "device_name": "CPU",
        "device_count": 1,
    }
    if torch.cuda.is_available():
        # Check for NVIDIA CUDA/AMD ROCm
        device_info["device"] = "cuda"
        device_info["device_count"] = torch.cuda.device_count()
        device_info["device_name"] = torch.cuda.get_device_name(0)
        # Distinguish between NVIDIA CUDA and AMD ROCm
        if torch.version.hip is not None:
            # AMD GPU
            device_info["backend"] = "rocm"
        else:
            # NVIDIA GPU
            device_info["backend"] = "cuda"
        return device_info
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check for Apple Silicon MPS
        device_info["device"] = "mps"
        device_info["backend"] = "mps"
        device_info["device_name"] = "Apple Silicon"
        return device_info
    else:
        # Check for TPU (requires torch_xla)
        try:
            import os
            import torch_xla.core.xla_model as xm
            # Hard signal: TPU runtime explicitly enabled
            if os.environ.get("PJRT_DEVICE") == "TPU":
                # forces runtime init
                _ = xm.xla_device()
                # best effort world size detection
                try:
                    count = xm.xrt_world_size()
                    if count == 0:
                        raise RuntimeError
                except Exception:
                    # fallback, visible XLA devices
                    visible = os.environ.get("XLA_VISIBLE_DEVICES")
                    if visible:
                        count = len(visible.split(","))
                    else:
                        # Single-process TPU (still valid)
                        count = 1
                device_info["device"] = "xla"
                device_info["backend"] = "tpu"
                device_info["device_name"] = "TPU"
                device_info["device_count"] = count
                return device_info
        except ImportError:
            pass

    return device_info


def setup_backend_settings(backend: str) -> None:
    """Configure backend-specific optimizations and seeding."""
    import torch

    # torch.manual_seed(3407) is all you need
    # https://arxiv.org/pdf/2109.08203
    seed = 3407
    torch.manual_seed(seed)
    if backend in ("cuda", "rocm"):
        _setup_cuda_backend(seed, backend)
    elif backend == "tpu":
        _setup_tpu_backend(seed)
    elif backend == "mps":
        _setup_mps_backend()
    # CPU needs no additional setup beyond torch.manual_seed()


def _setup_cuda_backend(seed: int, backend: str) -> None:
    """Configure CUDA/ROCm backend settings."""
    import torch

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if backend == "cuda":
        if torch.cuda.get_device_capability()[0] >= 8:
            # TF32 settings (NVIDIA Ampere+ only, compute capability >= 8.0)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()


def _setup_tpu_backend(seed: int) -> None:
    """Configure TPU backend settings."""
    import torch_xla.core.xla_model as xm

    xm.set_rng_state(seed)


def _setup_mps_backend() -> None:
    """Configure Apple MPS backend settings."""
    import torch

    # MPS uses torch.manual_seed() for seeding
    torch.mps.empty_cache()
