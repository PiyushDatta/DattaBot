import os
from sentencepiece import SentencePieceProcessor
from src.util import TOKENIZER_MODEL_PATH
from src.logger import get_logger
from src.agent_config import get_agent_config
from src.model import DattaBotModel


class Agent:
    def __init__(self) -> None:
        # Setup logger and config, both singletons.
        self.logger = get_logger()
        self.config = get_agent_config()
        # Setup tokenizer.
        assert os.path.isfile(TOKENIZER_MODEL_PATH), TOKENIZER_MODEL_PATH
        self.tokenizer = SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)
        self.logger.info(f"Loaded tokenizer model from path: {TOKENIZER_MODEL_PATH}")
        self.n_words: int = self.tokenizer.vocab_size()
        self.bos_id: int = self.tokenizer.bos_id()
        self.eos_id: int = self.tokenizer.eos_id()
        self.pad_id: int = self.tokenizer.pad_id()
        self.logger.debug(f"Tokenizer Words: {self.n_words}")
        self.logger.debug(f"Tokenizer BOS ID: {self.bos_id}")
        self.logger.debug(f"Tokenizer EOS ID: {self.eos_id}")
        # Setup model.
        self.model = DattaBotModel(config=self.config)

    def respond_to_query(self, query: str) -> str:
        encoded = self.tokenizer_encode(query=query)
        self.logger.info(f"Encoded {encoded}")
        return self.tokenizer_decode(encoded=encoded)

    def tokenizer_encode(self, query: str) -> list[int]:
        return self.tokenizer.encode(query)

    def tokenizer_decode(self, encoded: str) -> str:
        return self.tokenizer.decode(encoded)
