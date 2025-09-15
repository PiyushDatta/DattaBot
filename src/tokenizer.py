"""
Generic Tokenizer Wrapper (singleton)
Default: HarmonyTokenizer (o200k_harmony)
"""

import tiktoken
from typing import Optional, Union


class DattaBotTokenizer:
    _instance: Optional["DattaBotTokenizer"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, encoding_name: Optional[str] = None):
        # Prevent re-initialization on subsequent calls
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.name = encoding_name
        if self.name is None:
            self.name = "o200k_harmony"
            self._tokenizer = self._get_harmony_tokenizer()
        elif self.name == "o200k_harmony":
            self._tokenizer = self._get_harmony_tokenizer()
        else:
            self._tokenizer = tiktoken.get_encoding(encoding_name)

        # Add BOS, EOS, PAD as defaults
        self._bos_token = "<|startoftext|>"
        self._eos_token = "<|endoftext|>"
        # use EOS as PAD fallback
        self._pad_token = self._eos_token

        self._bos_token_id = self._tokenizer._special_tokens[self._bos_token]
        self._eos_token_id = self._tokenizer._special_tokens[self._eos_token]
        self._pad_token_id = self._tokenizer._special_tokens[self._pad_token]

        assert self._pad_token_id != -1, "Pad id can't be -1."

        self._initialized = True

    def _get_harmony_tokenizer(self):
        """Custom Harmony tokenizer based on o200k_base."""
        o200k_base = tiktoken.get_encoding("o200k_base")
        return tiktoken.Encoding(
            name="o200k_harmony",
            pat_str=o200k_base._pat_str,
            mergeable_ranks=o200k_base._mergeable_ranks,
            special_tokens={
                **o200k_base._special_tokens,
                "<|startoftext|>": 199998,
                "<|endoftext|>": 199999,
                "<|reserved_200000|>": 200000,
                "<|reserved_200001|>": 200001,
                "<|return|>": 200002,
                "<|constrain|>": 200003,
                "<|reserved_200004|>": 200004,
                "<|channel|>": 200005,
                "<|start|>": 200006,
                "<|end|>": 200007,
                "<|message|>": 200008,
                "<|reserved_200009|>": 200009,
                "<|reserved_200010|>": 200010,
                "<|reserved_200011|>": 200011,
                "<|call|>": 200012,
            }
            | {f"<|reserved_{i}|>": i for i in range(200013, 201088)},
        )

    # --- Public API ---
    def encode(
        self, text_or_texts: Union[str, list[str]]
    ) -> Union[list[int], list[list[int]]]:
        """
        Encode a single string or a list of strings.
        Automatically adds BOS/EOS tokens.
        Returns:
            - list[int] for a single string
            - list[list[int]] for a list of strings
        """
        if isinstance(text_or_texts, str):
            return (
                [self.bos_token_id]
                + self._tokenizer.encode(text_or_texts)
                + [self.eos_token_id]
            )
        elif isinstance(text_or_texts, list):
            return [
                [self.bos_token_id] + self._tokenizer.encode(text) + [self.eos_token_id]
                for text in text_or_texts
            ]
        else:
            raise TypeError(f"Expected str or list[str], got {type(text_or_texts)}")

    def decode(
        self, tokens_or_tokens_list: Union[list[int], list[list[int]]]
    ) -> Union[str, list[str]]:
        """
        Decode a single list of token IDs or a list of lists of token IDs.
        Returns:
            - str for a single list of token IDs
            - list[str] for a list of lists of token IDs
        """
        if isinstance(tokens_or_tokens_list, list) and all(
            isinstance(t, int) for t in tokens_or_tokens_list
        ):
            return self._tokenizer.decode(tokens_or_tokens_list)
        elif isinstance(tokens_or_tokens_list, list) and all(
            isinstance(t, list) for t in tokens_or_tokens_list
        ):
            return [self._tokenizer.decode(tokens) for tokens in tokens_or_tokens_list]
        else:
            raise TypeError(
                f"Expected list[int] or list[list[int]], got {type(tokens_or_tokens_list)}"
            )

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tokenizer._special_tokens.get(
            token
        ) or self._tokenizer._mergeable_ranks.get(token)

    def id_to_token(self, token_id: int) -> str:
        for k, v in self._tokenizer._special_tokens.items():
            if v == token_id:
                return k
        for k, v in self._tokenizer._mergeable_ranks.items():
            if v == token_id:
                return k
        raise ValueError(f"Token id {token_id} not found in vocabulary")

    # --- Token Properties ---

    @property
    def vocab(self) -> dict[str, int]:
        """
        Returns the full tokenizer vocabulary as a dictionary: token -> id
        Includes both special tokens and mergeable ranks.
        """
        return {**self._tokenizer._special_tokens, **self._tokenizer._mergeable_ranks}

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer._mergeable_ranks) + len(
            self._tokenizer._special_tokens
        )

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def backend(self):
        return self._tokenizer

    def __repr__(self):
        return (
            f"DattaBotTokenizer(name={self.name}, "
            f"vocab_size={self.vocab_size}, "
            f"bos_token={self.bos_token}:{self.bos_token_id}, "
            f"eos_token={self.eos_token}:{self.eos_token_id}, "
            f"pad_token={self.pad_token}:{self.pad_token_id})"
        )


# --- Singleton accessor ---
def get_tokenizer(encoding_name: Optional[str] = None) -> DattaBotTokenizer:
    """
    Get the singleton DattaBotTokenizer instance.
    """
    return DattaBotTokenizer(encoding_name=encoding_name)
