from omegaconf import DictConfig
from torch import (
    dtype as torch_dtype,
    float64 as torch_float64,
    float32 as torch_float32,
    float16 as torch_float16,
    int64 as torch_int64,
    int32 as torch_int32,
)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def get_tensor_dtype_from_config(config: DictConfig) -> torch_dtype:
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

    assert 0, f"Unsupported tensor dtype: {config_tensor_dtype}."
    return None
