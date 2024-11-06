from enum import IntEnum
from torch import Tensor


class AgentAction(IntEnum):
    # Always keep this as first in enum.
    NO_ACTION_START = 0
    GET_RESPONSES_FOR_QUERIES = 1
    GET_ENCODINGS_FOR_QUERIES = 2
    GET_DECODINGS_FOR_QUERIES = 3
    GET_ENCODED_TENSORS_FOR_QUERIES = 4
    TRAIN_AGENT = 5
    # Always keep this as last in enum.
    NO_ACTION_END = 6


class DattaBotAPIException(Exception):
    "Raised when the DattaBot API has an error."
    pass


class DattaBotAPIResponse:
    def __init__(self) -> None:
        # The response to the request queries in a string human read-able form.
        # Usually human language words.
        self._query_response: str = ""
        # The response to the queries in a Tensor form.
        self._tensor_response: Tensor = None
        # The number of training batches used.
        self._num_train_batches: int = 0
        # The number of validation batches used.
        self._num_val_batches: int = 0
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
        return self._num_train_batches + self._num_val_batches

    @property
    def num_train_batches(self) -> int:
        return self._num_train_batches

    @num_train_batches.setter
    def num_train_batches(self, value: int) -> None:
        self._num_train_batches = value

    @property
    def num_val_batches(self) -> int:
        return self._num_val_batches

    @num_val_batches.setter
    def num_val_batches(self, value: int) -> None:
        self._num_val_batches = value

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
