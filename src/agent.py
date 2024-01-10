from torch import Tensor
from sentencepiece import SentencePieceProcessor
from os.path import isfile

from src.logger import get_logger
from src.agent_config import get_agent_config
from src.logger import get_logger
from src.agent_config import get_agent_config
from src.data_loader import DataLoader
from src.model import DattaBotModel
from src.util import TOKENIZER_MODEL_PATH


class Agent:
    def __init__(self) -> None:
        # Setup config and logger, both singletons.
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        # Setup tokenizer.
        assert isfile(TOKENIZER_MODEL_PATH), TOKENIZER_MODEL_PATH
        self.tokenizer: SentencePieceProcessor = SentencePieceProcessor(
            model_file=TOKENIZER_MODEL_PATH
        )
        self.logger.info(f"Loaded tokenizer model from path: {TOKENIZER_MODEL_PATH}")
        # Setup data loader.
        self.data_loader = DataLoader(tokenizer=self.tokenizer)
        # Setup model.
        self.model = DattaBotModel(tokenizer=self.tokenizer)
        # Batch size.
        self.batch_size = self.config.agent.batch_size
        self.logger.debug(f"Batch size: {self.batch_size}")
        # Model dimensions.
        self.model_dimensions = self.config.neural_net.model_dimensions
        self.logger.debug(f"Model dimensions: {self.model_dimensions}")
        # Max tokens for response.
        self.response_max_tokens = self.config.agent.max_tokens
        self.logger.debug(f"Max tokens: {self.response_max_tokens}")

    def respond_to_queries(self, queries: list[str]) -> str:
        # Encode the list of queries and convert them into a tensors.
        # Tensor, int
        input_tensor, total_batches = self.convert_queries_to_tensors(queries=queries)
        self.logger.debug(
            f"Output of self.convert_queries_to_tensors():\n{input_tensor}\nwith shape: {input_tensor.shape}\nTotal batches: {total_batches}"
        )
        # Feed the tensors to our model, in batches.
        batched_tensor_responses = []
        total_loss = 0
        for batch, i in enumerate(range(0, total_batches)):
            data, targets = self.data_loader.get_batch(
                input_tensor=input_tensor, idx=i, train=False
            )
            # Call our model.
            self.logger.debug(
                f"Data being fed to model:\n{data}\nwith shape: {data.shape}"
            )
            output = self.model(data)
            # Flatten and add to our responses.
            # output_flat = output.view(-1, self.response_max_tokens)
            batched_tensor_responses.append(str(output))
        # Decode the tensors and return the string reprensation of the
        # agent's response.
        return batched_tensor_responses
        # return self.data_loader.convert_tensor_to_responses(
        #     tensors=batched_tensor_responses
        # )

    def tokenizer_encode(self, decoded_queries: list[str]) -> list[list[int]]:
        return self.data_loader.tokenizer_encode(decoded_queries=decoded_queries)

    def tokenizer_decode(self, encoded_queries: list[list[int]]) -> list[str]:
        return self.data_loader.tokenizer_decode(encoded_queries=encoded_queries)

    def convert_queries_to_tensors(self, queries: list[str]) -> tuple[Tensor, int]:
        return self.data_loader.convert_queries_to_tensors(queries=queries)
