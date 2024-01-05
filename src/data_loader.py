from torch import (
    tensor,
    Tensor,
    nn,
    float64 as torch_float64,
    stack,
    cat,
    empty as torch_empty_tensor,
)
from torch.nn.utils.rnn import pad_sequence
from sentencepiece import SentencePieceProcessor

from src.agent_config import get_agent_config
from src.logger import get_logger


class DataLoader:
    def __init__(self, tokenizer: SentencePieceProcessor) -> None:
        # Setup logger and config, both singletons.
        self.logger = get_logger()
        self.logger.debug(f"{self.__class__.__name__} init.")
        self.config = get_agent_config()
        self.tokenizer = tokenizer
        self.vocab_size: int = self.tokenizer.vocab_size()
        self.bos_id: int = self.tokenizer.bos_id()
        self.eos_id: int = self.tokenizer.eos_id()
        self.pad_id: int = self.tokenizer.pad_id()
        self.logger.debug(f"Tokenizer Words: {self.vocab_size}")
        self.logger.debug(f"Tokenizer BOS ID: {self.bos_id}")
        self.logger.debug(f"Tokenizer EOS ID: {self.eos_id}")
        # Block size.
        self.block_size = self.config.agent.input_block_size
        self.logger.debug(f"Block size: {self.block_size}")
        # Max sequence length.
        self.max_sequence_len = self.config.neural_net.model_dimensions
        self.logger.debug(f"Block size: {self.max_sequence_len}")
        # Batch size
        self.batch_size = self.config.agent.batch_size
        self.logger.debug(f"Batch size: {self.batch_size}")

    def convert_queries_to_tensors(self, queries: list[str]) -> tuple[Tensor, int]:
        """
        Returns tuple[Tensor, int].
        Returns tuple[queries converted to tensors, total number of batches].
        """
        # Encode the list of queries using the tokenizer.
        encodings: list[list[torch_float64]] = self.tokenizer_encode(
            decoded_queries=queries
        )
        self.logger.info(f"Encoded: {encodings}")
        # For each encoded query:
        #   1. Convert to a tensor.
        #   2. Pad until max_sequence_len length.
        padded_tensors: list[Tensor] = [
            nn.functional.pad(
                tensor(seq, dtype=torch_float64),
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

        # TODO(PiyushDatta): Do we need to filter empty tensors out?
        #                    Is this the right way to do it?
        # Filter the tensors for any empty tensors.
        # encoded_tensor = cat(tuple(filter(lambda t: t.numel() > 0, encoded_tensor)))
        # self.logger.debug(f"After filtering tensors: {encoded_tensor}")

        # Divide the tensors into batch_sized separate sequences.
        sequence_len = encoded_tensor.size(0)
        # Trim the tensors. Trim/get rid of the excess.
        # The code `seq_len * batch_size` represents the total number of
        # elements required for a perfect division into batches.
        batch_size = min(self.batch_size, encoded_tensor.size(1))
        total_number_of_elements = sequence_len * batch_size
        encoded_tensor = encoded_tensor[:total_number_of_elements]
        self.logger.debug(
            f"Unbatched tensors: {encoded_tensor}\nwith shape: {encoded_tensor.shape}"
        )
        # Reshape the tensors into a 2D Tensor (batch_size, sequence_len).
        encoded_tensor = encoded_tensor.view(batch_size, sequence_len)
        self.logger.debug(
            f"Batched tensors: {encoded_tensor}\nwith shape: {encoded_tensor.shape}"
        )
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
        encoded_tensor = encoded_tensor.t().contiguous()
        self.logger.debug(
            f"Encoded tensor inputs for the model:\n{encoded_tensor}\nwith shape: {encoded_tensor.shape}"
        )
        return encoded_tensor, total_number_of_elements // batch_size

    def convert_tensor_to_responses(self, tensor: list[Tensor]) -> list[str]:
        return tensor
        # Convert the Tensor from the model to a output string.
        # output: str = convert_tensor_output_to_str(encoded_tensor_output)
        # self.logger.info(f"Model output: {output}")
        # Return the output after decoding the output using the tokenizer.
        # return self.tokenizer_decode(encoded_queries=output)

    def get_batch(
        self, input_tensor: Tensor, idx: int, train: bool
    ) -> tuple[Tensor, Tensor]:
        seq_len = input_tensor.size(0)
        data = input_tensor[idx : idx + seq_len]
        target = torch_empty_tensor((seq_len, self.batch_size))
        if train:
            target = input_tensor[idx + 1 : idx + seq_len + 1].reshape(-1)
        return data, target

    def tokenizer_encode(self, decoded_queries: list[str]) -> list[list[int]]:
        return [self.tokenizer.encode(query) for query in decoded_queries]

    def tokenizer_decode(self, encoded_queries: list[list[int]]) -> list[str]:
        return [self.tokenizer.decode(query) for query in encoded_queries]
