import os
from sentencepiece import SentencePieceProcessor

MODEL_PATH: str = "tokenizer.model"


class Agent:
    def __init__(self) -> None:
        assert os.path.isfile(MODEL_PATH), MODEL_PATH
        self.tokenizer = SentencePieceProcessor(model_file=MODEL_PATH)
        print(f"Loaded tokenizer model from path: {MODEL_PATH}")
        self.n_words: int = self.tokenizer.vocab_size()
        self.bos_id: int = self.tokenizer.bos_id()
        self.eos_id: int = self.tokenizer.eos_id()
        self.pad_id: int = self.tokenizer.pad_id()
        print(f"Words: {self.n_words}\nBOS ID: {self.bos_id}\nEOS ID: {self.eos_id}")

    def respond_to_query(self, query: str) -> str:
        encoded = self.encode(query=query)
        print(f"Encoded {encoded}")
        return self.decode(encoded=encoded)

    def encode(self, query: str) -> list[int]:
        return self.tokenizer.encode(query)

    def decode(self, encoded: str) -> str:
        return self.tokenizer.decode(encoded)
