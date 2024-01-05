TOKENIZER_MODEL_PATH: str = "tokenizer.model"

from omegaconf import DictConfig
from torch import tensor, Tensor, float64


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def make_tensor_from_input(src_input: list[list[int]], config: DictConfig) -> Tensor:
    output_tensor = tensor(src_input, dtype=float64)
    model_dimensions = config.neural_net.model_dimensions
    return output_tensor.reshape(model_dimensions, model_dimensions)


def convert_tensor_output_to_str(src_tensor: Tensor) -> str:
    return src_tensor.detach().cpu().numpy()
