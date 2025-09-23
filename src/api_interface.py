from enum import IntEnum
from typing import Any, Optional

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
    """Raised when the DattaBot API has an error."""

    pass


from typing import Any, Optional

from torch import Tensor


class DattaBotAPIResponse:
    """
    Complete wrapper around OpenAI Harmony response.
    Separates raw Harmony payload (_response) from agent metadata (_metadata).
    """

    def __init__(
        self,
        response_dict: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        # Raw Harmony response
        self._response: dict[str, Any] = response_dict or {}
        # Local agent metadata
        self._metadata: dict[str, Any] = metadata or {}

    # -------------------------
    # Harmony response properties
    # -------------------------
    @property
    def text(self) -> str:
        return self._response.get("output_text", "")

    @text.setter
    def text(self, value: str):
        self._response["output_text"] = value

    @property
    def choices(self) -> list[dict[str, Any]]:
        return self._response.get("choices", [])

    @choices.setter
    def choices(self, value: list[dict[str, Any]]):
        self._response["choices"] = value

    @property
    def usage(self) -> dict[str, Any]:
        return self._response.get("usage", {})

    @usage.setter
    def usage(self, value: dict[str, Any]):
        self._response["usage"] = value

    @property
    def raw(self) -> dict[str, Any]:
        return self._response

    # -------------------------
    # Metadata properties
    # -------------------------
    @property
    def tensor_response(self) -> Optional[Tensor]:
        return self._metadata.get("tensor_response", None)

    @tensor_response.setter
    def tensor_response(self, value: Optional[Tensor]) -> None:
        self._metadata["tensor_response"] = value

    @property
    def num_train_batches(self) -> int:
        return self._metadata.get("num_train_batches", 0)

    @num_train_batches.setter
    def num_train_batches(self, value: Optional[int]) -> None:
        self._metadata["num_train_batches"] = value if value is not None else 0

    @property
    def num_val_batches(self) -> int:
        return self._metadata.get("num_val_batches", 0)

    @num_val_batches.setter
    def num_val_batches(self, value: Optional[int]) -> None:
        self._metadata["num_val_batches"] = value if value is not None else 0

    @property
    def num_train_tokens_processed(self) -> int:
        return self._metadata.get("num_train_tokens_processed", 0)

    @num_train_tokens_processed.setter
    def num_train_tokens_processed(self, value: Optional[int]) -> None:
        self._metadata["num_train_tokens_processed"] = value if value is not None else 0

    @property
    def num_val_tokens_processed(self) -> int:
        return self._metadata.get("num_val_tokens_processed", 0)

    @num_val_tokens_processed.setter
    def num_val_tokens_processed(self, value: Optional[int]) -> None:
        self._metadata["num_val_tokens_processed"] = value if value is not None else 0

    @property
    def tokenizer_encodings(self) -> list[list[int]]:
        return self._metadata.get("tokenizer_encodings", [])

    @tokenizer_encodings.setter
    def tokenizer_encodings(self, value: Optional[list[list[int]]]) -> None:
        self._metadata["tokenizer_encodings"] = value if value is not None else []

    @property
    def tokenizer_decodings(self) -> list[str]:
        return self._metadata.get("tokenizer_decodings", [])

    @tokenizer_decodings.setter
    def tokenizer_decodings(self, value: Optional[list[str]]) -> None:
        self._metadata["tokenizer_decodings"] = value if value is not None else []

    # -------------------------
    # Debug / string representation
    # -------------------------
    def __str__(self) -> str:
        return (
            f"Query Response: {self.text}\n"
            f"Usage: {self.usage}\n"
            f"Raw: {self.raw}\n"
            f"Metadata: {self._metadata}"
        )

    def __repr__(self) -> str:
        return f"DattaBotAPIResponse(text={self.text!r}, usage={self.usage}, metadata={self._metadata})"
