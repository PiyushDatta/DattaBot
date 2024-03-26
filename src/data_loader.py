from torch import (
    tensor,
    Tensor,
    nn,
    cat,
    dtype as torch_dtype,
)
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from src.agent_config import get_agent_config
from src.logger import get_logger
from src.util import DattaBotAPIResponse, get_tensor_dtype_from_config


class DataLoader:
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        # Setup config and logger, both singletons.
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        self.logger.debug(f"{self.__class__.__name__} init.")
        self._tensor_dtype = get_tensor_dtype_from_config(config=self.config)
        self.tokenizer = tokenizer
        self.vocab_size: int = self.tokenizer.vocab_size
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id
        self.pad_id: int = self.tokenizer.pad_token_id
        # Things go bad when pad_id is -1, we can't even print an encoded tensor!
        assert self.pad_id != -1, f"Pad id can't be -1."
        self.logger.debug(f"Tokenizer Words: {self.vocab_size}")
        self.logger.debug(f"Tokenizer BOS ID: {self.bos_id}")
        self.logger.debug(f"Tokenizer EOS ID: {self.eos_id}")
        self.logger.debug(f"Tokenizer PAD ID: {self.pad_id}")
        # Block size.
        self.block_size = self.config.agent.input_block_size
        self.logger.debug(f"Block size: {self.block_size}")
        # Max sequence length.
        self.max_sequence_len = self.config.neural_net.model_dimensions
        self.logger.debug(f"Block size: {self.max_sequence_len}")
        # Batch size
        self.batch_size = self.config.agent.batch_size
        self.logger.debug(f"Batch size: {self.batch_size}")

    @property
    def tensor_dtype(self) -> torch_dtype:
        return self._tensor_dtype

    @tensor_dtype.setter
    def tensor_dtype(self, value: torch_dtype) -> None:
        self._tensor_dtype = value

    def convert_queries_to_tensors(self, queries: list[str]) -> tuple[Tensor, int]:
        """
        Returns tuple[Tensor, int].
        Returns tuple[queries converted to tensors, total number of batches].
        """
        self.logger.debug(f"\nQueries:\n{queries}")
        # Encode the list of queries using the tokenizer.
        encodings: list[list[self.tensor_dtype]] = self.tokenizer_encode(
            decoded_queries=queries
        )
        self.logger.debug(f"\nEncoded queries:\n{encodings}")
        # For each encoded query:
        #   1. Convert to a tensor.
        #   2. Pad until max_sequence_len length.
        padded_tensors: list[Tensor] = [
            nn.functional.pad(
                tensor(seq, dtype=self.tensor_dtype),
                (0, self.max_sequence_len - len(seq)),
                mode="constant",
                value=self.pad_id,
            )
            for seq in encodings
        ]
        self.logger.debug(f"Padded tensors:\n{padded_tensors}")
        # Now we combine all the tensors from padded_tensors into 1 tensor
        # with shape (len(queries), max_sequence_len).
        encoded_tensor: Tensor = pad_sequence(
            [query_tensor for query_tensor in padded_tensors],
            batch_first=True,
            padding_value=self.pad_id,
        )
        self.logger.debug(
            f"Encoded tensor:\n{encoded_tensor}\nwith shape: {encoded_tensor.shape}"
        )
        # Record the total batch size and sequence length before the next step
        # (combining into 1 tensor and filtering).
        total_batch_size = encoded_tensor.size(0)
        sequence_len = encoded_tensor.size(1)
        # Flatten the tensor into a single dimension, so combine all queries of
        # max_sequence_len into 1 query of
        # size = number of queries * max_sequence_len.
        # Also filter for any empty tensors.
        encoded_tensor = cat(tuple(filter(lambda t: t.numel() > 0, encoded_tensor)))
        # Divide the tensor into batch_sized separate sequences.
        # Trim the tensor. Trim/get rid of the excess.
        # The code `seq_len * batch_size` represents the total number of
        # elements required for a perfect division into batches.
        batch_size = min(total_batch_size, self.batch_size)
        total_number_of_elements = sequence_len * min(total_batch_size, batch_size)
        encoded_tensor = encoded_tensor[:total_number_of_elements]
        self.logger.debug(
            f"Flattened filtered unbatched tensors:\n{encoded_tensor}\nwith shape: {encoded_tensor.shape}\ntotal number of elements: {total_number_of_elements}"
        )
        # Reshape the tensors into a 2D Tensor (batch_size, sequence_len).
        encoded_tensor = encoded_tensor.view(batch_size, sequence_len).contiguous()
        self.logger.debug(
            f"Batched tensors: {encoded_tensor}\nwith shape: {encoded_tensor.shape}"
        )
        # TODO(PiyushDatta): Do we need the below? Our layers are happy to take
        #                    batch_size, seq_len.
        #
        # For some LSTM layers they may take the order of [sequence_len, batch_size].
        # Even though, from our view we can see that we put the order as
        # [batch_size, sequence_len] since it a natural way to align the data.
        # Each row/batch is the sequence. So we simply need to transpose the tensors,
        # or more simply put, just swap the ordering to be [sequence_len, batch_size]
        # to satisfy the ordering of the layer.
        #
        # Also maybe batch as second dimension is faster on Nvidia CUDA if
        # batch_size is second?
        # https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch/49473068#49473068
        #
        # Calling transpose() does not gaurantee contiguous memory layout of tensor.
        # So call contiguous() to make sure.
        # encoded_tensor = encoded_tensor.t().contiguous()
        self.logger.debug(
            f"Encoded tensor inputs for the model:\n{encoded_tensor}\nwith shape: {encoded_tensor.shape}"
        )
        return encoded_tensor, total_number_of_elements // batch_size

    def convert_tensors_to_responses(
        self, tensor_resps: list[Tensor]
    ) -> DattaBotAPIResponse:
        response: DattaBotAPIResponse = DattaBotAPIResponse()
        response.tensor_response = cat(tensor_resps, dim=-1)
        decoded_responses = []
        for response_tensor in tensor_resps:
            decoded_output = self.tokenizer.decode(
                response_tensor[0], skip_special_tokens=True
            )
            decoded_responses.append(decoded_output)

        response.query_response = " ".join(decoded_responses)
        return response

    def get_batch(
        self, input_tensor: Tensor, idx: int, train: bool
    ) -> tuple[Tensor, Tensor]:
        seq_len = input_tensor.size(0)
        data = input_tensor[idx : idx + seq_len]
        # target = torch_empty_tensor((seq_len, self.batch_size))
        # if train:
        target = input_tensor[idx + 1 : idx + seq_len + 1].reshape(-1)
        return data, target

    def tokenizer_encode(self, decoded_queries: list[str]) -> list[list[int]]:
        return [
            self.tokenizer.encode_plus(
                query,
                add_special_tokens=True,
            )["input_ids"]
            for query in decoded_queries
        ]

    def tokenizer_decode(self, encoded_queries: list[list[int]]) -> list[str]:
        return [self.tokenizer.decode(query) for query in encoded_queries]

    # TODO
    def get_formatted_batched_training_data():
        pass
