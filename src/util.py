from omegaconf import DictConfig
from torch import (
    Tensor,
    dtype as torch_dtype,
    float64 as torch_float64,
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

    assert 0, f"Unsupported tensor dtype: {config_tensor_dtype}."
    return None


class DattaBotAPIResponse:
    def __init__(self) -> None:
        # The response to the request queries in a string human read-able form.
        # Usually human language words.
        self._query_response: str = ""
        # The response to the queries in a Tensor form.
        self._tensor_response: Tensor = None
        # The number of batches used.
        self._num_batches: int = None
        # Tokenizer encodings of the request queries.
        self._tokenizer_encodings: list[list[int]] = []
        # Tokenizer decodings of the request queries.
        self._tokenizer_decodings: list[str] = []

    # Add a __str__ method for a pretty print representation
    def __str__(self) -> str:
        response_str = f"Query Response: {self.query_response}\n"
        response_str += f"Number of Batches: {self.num_batches}\n"
        if self.tensor_response is not None:
            response_str += f"Tensor Response: {self.tensor_response}\n"
            response_str += f"Tensor Response Shape: {self.tensor_response.shape}\n"

        if self.tokenizer_encodings:
            response_str += "Tokenizer Encodings:\n"
            for encoding in self.tokenizer_encodings:
                response_str += f"  {encoding}\n"

        if self.tokenizer_decodings:
            response_str += "Tokenizer Decodings:\n"
            for decoding in self.tokenizer_decodings:
                response_str += f"  {decoding}\n"
        return response_str

    @property
    def query_response(self) -> str:
        return self._query_response

    @query_response.setter
    def query_response(self, value: str) -> None:
        self._query_response = value

    @property
    def tensor_response(self) -> Tensor:
        return self._tensor_response

    @tensor_response.setter
    def tensor_response(self, value: Tensor) -> None:
        self._tensor_response = value

    @property
    def num_batches(self) -> int:
        return self._num_batches

    @num_batches.setter
    def num_batches(self, value: int) -> None:
        self._num_batches = value

    @property
    def tokenizer_encodings(self) -> list[list[int]]:
        return self._tokenizer_encodings

    @tokenizer_encodings.setter
    def tokenizer_encodings(self, value: list[list[int]]) -> None:
        self._tokenizer_encodings = value

    @property
    def tokenizer_decodings(self) -> list[str]:
        return self._tokenizer_decodings

    @tokenizer_decodings.setter
    def tokenizer_decodings(self, value: list[str]) -> None:
        self._tokenizer_decodings = value
