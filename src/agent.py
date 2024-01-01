import os
from sentencepiece import SentencePieceProcessor
from torch import Tensor
from src.util import (
    TOKENIZER_MODEL_PATH,
    convert_tensor_output_to_str,
    make_tensor_from_input,
)
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
        self.vocab_size: int = self.tokenizer.vocab_size()
        self.bos_id: int = self.tokenizer.bos_id()
        self.eos_id: int = self.tokenizer.eos_id()
        self.pad_id: int = self.tokenizer.pad_id()
        self.logger.debug(f"Tokenizer Words: {self.vocab_size}")
        self.logger.debug(f"Tokenizer BOS ID: {self.bos_id}")
        self.logger.debug(f"Tokenizer EOS ID: {self.eos_id}")
        # Setup model.
        self.model = DattaBotModel(tokenizer=self.tokenizer)

    def respond_to_queries(self, queries: list[str]) -> str:
        # Encode the list of queries using the tokenizer.
        encodings: list[list[int]] = self.tokenizer_encode(queries=queries)
        self.logger.info(f"Encoded: {encodings}")
        # Convert the query string to a Tensor and feed into our model.
        encoded_tensor_input: Tensor = make_tensor_from_input(
            src_input=encodings, config=self.config
        )
        self.logger.info(f"Encoded tensor inputs for the model: {encoded_tensor_input}")
        encoded_tensor_output: Tensor = self.model(src_input=encoded_tensor_input)
        # Convert the Tensor from the model to a output string.
        output: str = convert_tensor_output_to_str(encoded_tensor_output)
        self.logger.info(f"Model output: {output}")
        # Return the output after decoding the output using the tokenizer.
        return self.tokenizer_decode(encoded=output)

    def tokenizer_encode(self, queries: list[str]) -> list[list[int]]:
        return [self.tokenizer.encode(query) for query in queries]

    def tokenizer_decode(self, encoded: str) -> str:
        return self.tokenizer.decode(encoded)
