from enum import Enum
import os
from typing import Any, Optional
import torch
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from torch import (
    tensor,
    Tensor,
    nn,
    cat,
    dtype as torch_dtype,
)
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer
from src.agent_config import get_agent_config
from src.logger import get_logger
from src.api_interface import DattaBotAPIResponse
from src.util import get_tensor_dtype_from_config


# Define the enum for dataset selection
class DatasetType(Enum):
    OPENWEBTEXT = "openwebtext"
    AG_NEWS = "ag_news"
    WIKITEXT = "wikitext"


def dataset_has_save_mechanism_feature(dataset: DatasetType):
    return dataset in (DatasetType.OPENWEBTEXT)


# Function to convert a string to a DatasetType enum
def string_to_enum(dataset_name: str) -> DatasetType:
    try:
        return DatasetType[dataset_name.upper()]
    except KeyError:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. Available options are: {list(DatasetType._member_names_)}"
        )


class DattabotDataLoader(TorchDataLoader):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_dir: str = "./dattabot_data_dir",
        dataset_name: str = "ERROR_DID_NOT_INSERT_DATASET_NAME",
        dataset: Optional[Dataset] = None,
        batch_size=32,
        reset_batch_when_reach_end=True,
        **kwargs: Any,
    ) -> None:
        # Setup config and logger, both singletons.
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        self.logger.debug(f"{self.__class__.__name__} init.")
        # Store tokenizer-related attributes
        self.tokenizer = tokenizer
        self.vocab_size: int = self.tokenizer.vocab_size
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id
        self.pad_id: int = self.tokenizer.pad_token_id
        assert self.pad_id != -1, "Pad id can't be -1."
        self.data_dir = data_dir
        self.batch_size = batch_size
        self._tensor_dtype = get_tensor_dtype_from_config(config=self.config)
        self._length = 0
        self.reset_batch_end = reset_batch_when_reach_end
        # If a dataset is provided, initialize the parent TorchDataLoader
        self.dataset_name = dataset_name
        self._dataset = dataset
        if dataset is not None:
            self.logger.info("Preparing dataset...")
            super().__init__(dataset=dataset, batch_size=self.batch_size, **kwargs)
            self._length = len(dataset) // batch_size + (
                1 if len(dataset) % batch_size != 0 else 0
            )
        # Initialize the iterators
        self.logger.info(f"Setting up iterator for {__class__}")
        self._data_iterator = self._get_iter()
        self.logger.info(f"Done setting up iterator for {__class__}!")

    def __len__(self) -> int:
        """Return the number of batches in the dataloader."""
        return self._length

    def __next__(self) -> Any:
        """Get the next batch in iteration, with optional reset."""
        try:
            # Retrieve the next batch from the current iterator
            batch = next(self._data_iterator)
        except StopIteration:
            self.logger.debug(
                f"While getting next iterator, reached end of data iterator. StopIteration. Value of self.reset_batch_end: {self.reset_batch_end}"
            )
            if self.reset_batch_end:
                self._reset()
                batch = next(self._data_iterator)
            else:
                raise StopIteration
        return batch

    def _reset(self):
        self._data_iterator = self._get_iter()

    def _get_iter(self):
        return super().__iter__()


class DattabotDataBuilder:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        **kwargs: Any,
    ) -> None:
        # Setup config and logger, both singletons.
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        self.logger.debug(f"{self.__class__.__name__} init.")
        # Tokenizer
        self.tokenizer = tokenizer
        self._tensor_dtype = get_tensor_dtype_from_config(config=self.config)
        # Batch size
        self.batch_size = self.config.agent.batch_size
        self.logger.debug(f"Batch size: {self.batch_size}")
        self.data_dir = self.config.agent.data_directory
        self.dataset_name: DatasetType = string_to_enum(
            dataset_name=self.config.env.dataset_name
        )

    # Build vocabulary function with progress display
    def build_vocab_with_progress(self) -> dict[str, int]:
        # TODO(PiyushDatta): See below todo (worth training our own tokenizer vocab).
        # Wrapper function to yield tokens with progress bar
        # def yield_tokens(data_lst):
        #     for text in tqdm(
        #         data_lst, desc="Tokenizing and building Vocabulary", total=total_items
        #     ):
        #         yield self.tokenizer.tokenize(text)

        # Build vocabulary from token iterator
        self.logger.info("Building the vocabulary...")
        vocab: dict[str, int] = self.tokenizer.get_vocab()
        vocab_size = self.tokenizer.vocab_size
        self.logger.info(f"Using pretrained tokenizer vocabulary size: {vocab_size}")
        # TODO(PiyushDatta): See if it is worth training our own tokenizer vocab.
        # vocab = build_vocab_from_iterator(
        #     yield_tokens(data_lst), specials=["<unk>", "<pad>"]
        # )
        # vocab.set_default_index(vocab["<unk>"])
        self.logger.info("Done building the vocabulary...")
        return vocab

    def download_dataset(self):
        """
        Download and return the appropriate train and validation data
        """
        match self.dataset_name:
            case DatasetType.AG_NEWS:
                dataset = load_dataset(self.dataset_name.value, trust_remote_code=True)
                train_data = dataset["train"]
                validation_data = dataset["test"]
            case DatasetType.OPENWEBTEXT:
                dataset = load_dataset(self.dataset_name.value, trust_remote_code=True)
                # Openwebtext only have "train" as a key, so we split 80/20 here.
                split_dataset = dataset["train"].train_test_split(test_size=0.2)
                train_data = split_dataset["train"]
                validation_data = split_dataset["test"]
            case DatasetType.WIKITEXT:
                # WikiText-103 specifically.
                dataset = load_dataset(
                    "wikitext", "wikitext-103-raw-v1", trust_remote_code=True
                )
                train_data = dataset["train"]
                validation_data = dataset["validation"]
            case _:
                raise ValueError(f"Unrecognized dataset type: {self.dataset_name}")
        self.logger.info(
            f"{self.dataset_name.name} dataset loaded with train and validation splits."
        )
        return train_data, validation_data

    def setup_data(self):
        """
        Download and setup dataset if not already present
        """
        # Create data directory.
        self.logger.info("Setting up directories")
        os.makedirs(self.data_dir, exist_ok=True)
        # Define paths based on dataset name
        train_file = os.path.join(self.data_dir, f"{self.dataset_name}_train.pt")
        val_file = os.path.join(self.data_dir, f"{self.dataset_name}_val.pt")
        vocab_file = os.path.join(self.data_dir, f"{self.dataset_name}_vocab.pt")
        raw_train_path = os.path.join(
            self.data_dir, f"{self.dataset_name}_train_raw.pt"
        )
        raw_val_path = os.path.join(self.data_dir, f"{self.dataset_name}_val_raw.pt")

        # Try to load cached data
        try:
            if os.path.exists(train_file):
                self.logger.info("Loading cached dataset...")
                train_data = torch.load(train_file)
                val_data = torch.load(val_file)
                vocab = torch.load(vocab_file)
            else:
                # Check if raw data exists; if not, download and save it
                if os.path.exists(raw_train_path) and os.path.exists(raw_val_path):
                    self.logger.info("Loading raw data from cached files...")
                    train_data = torch.load(raw_train_path)
                    val_data = torch.load(raw_val_path)
                else:
                    self.logger.info(f"Downloading {self.dataset_name} dataset...")
                    train_iter, val_iter = self.download_dataset()
                    train_data, val_data = train_iter, val_iter
                    self.logger.info(
                        f"Saving data ({self.dataset_name}) so we don't have to download again."
                    )
                    torch.save(train_data, raw_train_path)
                    torch.save(val_data, raw_val_path)
                    self.logger.info("Raw data saved successfully.")
                    self.logger.info(
                        f"Finished downloading {self.dataset_name} dataset!"
                    )
                # Build vocabulary with progress tracking.
                self.logger.info(
                    f"Start building vocab from {self.dataset_name} dataset..."
                )
                vocab: dict[str, int] = self.build_vocab_with_progress()
                self.logger.info(
                    f"Done building vocab from {self.dataset_name} dataset!"
                )

            # Create datasets
            self.logger.info(f"Loading into our custom datasets.")
            seq_len = self.config.agent.max_response_tokens
            train_dataset = TextDataset(train_data, self.tokenizer, vocab, seq_len)
            val_dataset = TextDataset(val_data, self.tokenizer, vocab, seq_len)
            self.logger.info(f"Done loading into a custom dataset.")
            # Create dataloaders
            self.logger.info(f"Loading our training custom torch dataloader.")
            train_dataloader = DattabotDataLoader(
                tokenizer=self.tokenizer,
                dataset_name=self.dataset_name,
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=os.cpu_count(),
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
                generator=torch.Generator(device=self.config.env.device),
                reset_batch_when_reach_end=True,
            )
            self.logger.info(f"Loading our validation custom torch dataloader.")
            val_dataloader = DattabotDataLoader(
                tokenizer=self.tokenizer,
                dataset_name=self.dataset_name,
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=os.cpu_count(),
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
                generator=torch.Generator(device=self.config.env.device),
                reset_batch_when_reach_end=True,
            )
        except Exception as e:
            self.logger.error(f"Error setting up data: {str(e)}")
            raise
        self.logger.info(
            f"Done loading custom torch dataloaders! Done setting up data."
        )
        return train_dataloader, val_dataloader, vocab

    @property
    def tensor_dtype(self) -> torch.dtype:
        return self._tensor_dtype

    @tensor_dtype.setter
    def tensor_dtype(self, value: torch.dtype) -> None:
        self._tensor_dtype = value

    def convert_queries_to_tensors(
        self, queries: list[str], max_sequence_len: int
    ) -> tuple[Tensor, int]:
        """
        Returns tuple[Tensor, int].
        Returns tuple[queries converted to tensors, total number of batches].
        """
        # Encode the list of queries using the tokenizer
        encodings: list[list[int]] = self.tokenizer_encode(decoded_queries=queries)

        # For each encoded query: Convert to tensor and pad
        padded_tensors: list[Tensor] = [
            nn.functional.pad(
                tensor(seq, dtype=self.tensor_dtype),
                (0, max_sequence_len - len(seq)),
                mode="constant",
                value=self.pad_id,
            )
            for seq in encodings
        ]

        # Combine all tensors into one with shape (len(queries), max_sequence_len)
        encoded_tensor: Tensor = pad_sequence(
            [query_tensor for query_tensor in padded_tensors],
            batch_first=True,
            padding_value=self.pad_id,
        )

        # Record dimensions before processing
        total_batch_size = encoded_tensor.size(0)
        sequence_len = encoded_tensor.size(1)

        # Flatten and filter empty tensors
        encoded_tensor = cat(tuple(filter(lambda t: t.numel() > 0, encoded_tensor)))

        # Process into batches
        batch_size = min(total_batch_size, self.batch_size)
        total_number_of_elements = sequence_len * min(total_batch_size, batch_size)
        encoded_tensor = encoded_tensor[:total_number_of_elements]

        # Reshape into (batch_size, sequence_len)
        encoded_tensor = encoded_tensor.view(batch_size, sequence_len).contiguous()

        return encoded_tensor, total_number_of_elements // batch_size

    def convert_tensors_to_responses(self, tensor_resps: list[Tensor]) -> dict:
        if not tensor_resps:
            return {"query_response": "No responses generated."}

        # Concatenate tensors along last dimension
        tensor_response = torch.cat(tensor_resps, dim=0)

        # Convert to float if needed
        if self.tensor_dtype != torch.float:
            tensor_response = tensor_response.float()

        # Apply softmax and get predicted indices
        softmax_output = nn.functional.softmax(tensor_response, dim=-1)
        predicted_indices = torch.argmax(softmax_output, dim=-1)

        # Decode responses
        decoded_responses = []
        for response_tensor in predicted_indices:
            decoded_output = self.tokenizer.decode(
                response_tensor.tolist(),
                skip_special_tokens=True,
            )
            decoded_responses.append(decoded_output)

        return {
            "query_response": " ".join(decoded_responses),
            "tensor_response": tensor_response,
        }

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


class TextDataset(Dataset):
    def __init__(
        self, data: list[str], tokenizer: AutoTokenizer, vocab, seq_length: int = 256
    ):
        self.data = data
        self.vocab = vocab
        self.seq_length = seq_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get text and tokenize.
        item = self.data[idx]
        text = item["text"] if isinstance(item, dict) else item
        # Tokenize the text properly using the tokenizer.
        encoding = self.tokenizer(
            text,
            # +1 for target shifting
            max_length=self.seq_length + 1,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Get input_ids and remove the batch dimension added by tokenizer.
        tokens = encoding["input_ids"].squeeze(0)
        # Create input sequence (all tokens except last)
        x = tokens[:-1]
        # Create target sequence (all tokens except first)
        y = tokens[1:]
        # If somehow the sequence is shorter than seq_length (shouldn't happen due to padding)
        if x.size(0) < self.seq_length:
            padding_length = self.seq_length - x.size(0)
            x = torch.nn.functional.pad(
                x, (0, padding_length), value=self.tokenizer.pad_token_id
            )
            y = torch.nn.functional.pad(
                y, (0, padding_length), value=self.tokenizer.pad_token_id
            )

        return x, y
