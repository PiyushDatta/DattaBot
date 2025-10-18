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
    """Initialize torch distributed - imports only when called"""
    import torch.distributed as dist
    from torch.cuda import device_count as torch_device_count

    if not dist.is_available() or dist.is_initialized():
        return
    # Detect if we're running under torchrun or another distributed launcher.
    torchrun_set: bool = torch_device_count() > 1
    if not torchrun_set:
        print(
            "[setup_torch_dist_init] No torchrun environment detected, running single process."
        )
        return
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        # 60 minutes.
        timeout=timedelta(seconds=3600),
    )
    print(
        f"[setup_torch_dist_init] Initialized distributed (rank={dist.get_rank()}, world_size={dist.get_world_size()})"
    )
