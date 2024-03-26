from torch import Tensor, nn
from transformers import AutoTokenizer
from os.path import isfile

from src.logger import get_logger
from src.agent_config import get_agent_config
from src.logger import get_logger
from src.agent_config import get_agent_config
from src.data_loader import DataLoader
from src.model import DattaBotModel
from src.util import DattaBotAPIResponse, get_tensor_dtype_from_config


class Agent:
    def __init__(self) -> None:
        # Setup config and logger, both singletons.
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        # Setup tokenizer.
        tokenizer_model_name = "distilbert-base-uncased"
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_name
        )
        # Add bos and eos tokens and ids
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token
        self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.bos_token
        )
        self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.eos_token
        )
        # Things go bad when pad_id is -1, we can't even print an encoded tensor!
        # Pad token is -1, change it to eos token.
        assert self.tokenizer.pad_token_id != -1, f"Pad id can't be -1."
        self.logger.info(f"Loaded tokenizer model from path: {tokenizer_model_name}")
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
        self.response_max_response_tokens = self.config.agent.max_response_tokens
        self.logger.debug(f"Max tokens: {self.response_max_response_tokens}")

    def train_agent(self) -> DattaBotAPIResponse:
        self.model.train()
        epoch_loss = 0
        response: DattaBotAPIResponse = DattaBotAPIResponse()
        response.query_response = "TODO(PIYUSHDATTA): Fix me. Training agent."
        return response

    def respond_to_queries(self, queries: list[str]) -> DattaBotAPIResponse:
        # Encode the list of queries and convert them into a tensors.
        # Tensor, int
        input_tensor, total_batches = self.convert_queries_to_tensors(queries=queries)
        self.logger.debug(
            f"Output of self.convert_queries_to_tensors():\n{input_tensor}\nwith shape: {input_tensor.shape}\nTotal batches: {total_batches}"
        )
        # Feed the tensors to our model, in batches.
        batched_tensor_responses = []
        tensor_dtype = get_tensor_dtype_from_config(config=self.config)
        total_loss = 0
        for batch, i in enumerate(range(0, total_batches)):
            data, targets = self.data_loader.get_batch(
                input_tensor=input_tensor, idx=i, train=False
            )
            self.logger.debug(
                f"Data being fed to model:\n{data}\nwith shape: {data.shape}"
            )
            # Call our model.
            output = self.model(src_input=data, src_mask=None)
            # Apply softmax to convert logits to probabilities.
            softmax_output = nn.functional.softmax(output, dim=-1).to(tensor_dtype)
            # Flatten and add to our responses.
            output_flat = softmax_output.view(-1, self.response_max_response_tokens)
            batched_tensor_responses.append(output_flat)
            break
        # Decode the tensors and return a DattaBot API response for each tensor.
        response: DattaBotAPIResponse = self.convert_tensors_to_responses(
            tensor_resps=batched_tensor_responses
        )
        self.logger.debug(f"Output of self.respond_to_queries():\n{response}")
        return response

    def tokenizer_encode(self, decoded_queries: list[str]) -> list[list[int]]:
        return self.data_loader.tokenizer_encode(decoded_queries=decoded_queries)

    def tokenizer_decode(self, encoded_queries: list[list[int]]) -> list[str]:
        return self.data_loader.tokenizer_decode(encoded_queries=encoded_queries)

    def convert_queries_to_tensors(self, queries: list[str]) -> tuple[Tensor, int]:
        return self.data_loader.convert_queries_to_tensors(queries=queries)

    def convert_tensors_to_responses(
        self, tensor_resps: list[Tensor]
    ) -> DattaBotAPIResponse:
        return self.data_loader.convert_tensors_to_responses(tensor_resps=tensor_resps)
