import os
from sentencepiece import SentencePieceProcessor
from src.util import TOKENIZER_MODEL_PATH, get_default_logger


class Agent:
    def __init__(self) -> None:
        self.logger = get_default_logger()
        assert os.path.isfile(TOKENIZER_MODEL_PATH), TOKENIZER_MODEL_PATH
        self.tokenizer = SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)
        self.logger.info(f"Loaded tokenizer model from path: {TOKENIZER_MODEL_PATH}")
        self.n_words: int = self.tokenizer.vocab_size()
        self.bos_id: int = self.tokenizer.bos_id()
        self.eos_id: int = self.tokenizer.eos_id()
        self.pad_id: int = self.tokenizer.pad_id()
        self.logger.info(f"Words: {self.n_words}")
        self.logger.info(f"BOS ID: {self.bos_id}")
        self.logger.info(f"EOS ID: {self.eos_id}")

    def respond_to_query(self, query: str) -> str:
        encoded = self.encode(query=query)
        self.logger.info(f"Encoded {encoded}")
        return self.decode(encoded=encoded)

    def encode(self, query: str) -> list[int]:
        return self.tokenizer.encode(query)

    def decode(self, encoded: str) -> str:
        return self.tokenizer.decode(encoded)
