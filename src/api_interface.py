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


class DattaBotAPIResponse:
    """
    Complete wrapper around OpenAI Harmony response.
    """

    def __init__(self, query_response: Optional[dict[str, Any]] = None):
        self._response: dict[str, Any] = query_response or {}
        # Local agent fields (default None until set)
        self.tensor_response: Optional[Tensor] = None
        self._num_train_batches: Optional[int] = None
        self.tokenizer_encodings: Optional[list[list[int]]] = None
        self.tokenizer_decodings: Optional[list[str]] = None

    @property
    def text(self) -> str:
        """The main human-readable response."""
        return self._response.get("output_text", "") or ""

    @property
    def choices(self) -> Optional[list[dict[str, Any]]]:
        """Return the list of choices from the OpenAI response, if any."""
        return self._response.get("choices", None)

    @property
    def usage(self) -> Optional[dict[str, Any]]:
        """Return token usage info from the OpenAI response."""
        return self._response.get("usage", None)

    @property
    def raw(self) -> dict[str, Any]:
        """Return the original OpenAI response object."""
        return self._response

    def __str__(self) -> str:
        return f"Query Response: {self.text}\nUsage: {self.usage}\nRaw: {self.raw}"
