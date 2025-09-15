import os
from enum import Enum
from typing import Iterator, Optional, Union

import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader, Dataset

from src.agent_config import get_agent_config
from src.logger import get_logger
from src.tokenizer import get_tokenizer, DattaBotTokenizer


class DatasetType(Enum):
    OPENWEBTEXT = "openwebtext"
    AG_NEWS = "ag_news"
    WIKITEXT = "wikitext"


def string_to_enum(dataset_name: str) -> DatasetType:
    try:
        return DatasetType[dataset_name.upper()]
    except KeyError:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. Options: {list(DatasetType._member_names_)}"
        )


class TextDataset(Dataset):
    """Custom dataset for GPT-style training"""

    def __init__(
        self,
        data: list[Union[str, dict]],
        tokenizer: DattaBotTokenizer,
        seq_length: int = 256,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        item = self.data[idx]
        text = item["text"] if isinstance(item, dict) else item

        # Encode text
        tokens = self.tokenizer.encode(text)
        # Truncate or pad to seq_length + 1
        if len(tokens) < self.seq_length + 1:
            tokens += [self.tokenizer.pad_token_id] * (
                (self.seq_length + 1) - len(tokens)
            )
        else:
            tokens = tokens[: self.seq_length + 1]

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class DattabotDataLoader(TorchDataLoader):
    def __init__(
        self,
        dataset: Dataset,
        dataset_name: str,
        batch_size: int = 32,
        reset_batch_when_reach_end: bool = True,
        **kwargs,
    ):
        self._reset_batch_end = reset_batch_when_reach_end
        self.dataset_name = dataset_name
        self._dataset = dataset
        self._iterator: Optional[Iterator] = None
        super().__init__(dataset, batch_size=batch_size, **kwargs)

    def __iter__(self) -> Iterator:
        self._iterator = super().__iter__()
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        try:
            return next(self._iterator)
        except StopIteration:
            if self._reset_batch_end:
                self._iterator = super().__iter__()
                return next(self._iterator)
            else:
                raise


class DattabotDataBuilder:
    def __init__(self):
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        self.tokenizer = get_tokenizer()
        self.batch_size = self.config.agent.batch_size
        self.seq_len = self.config.agent.max_response_tokens
        self.data_dir = self.config.agent.data_directory
        self.dataset_name = string_to_enum(self.config.env.dataset_name)
        self.agent_device = self.config.env.device

    def download_dataset(self):
        """Download dataset and split into train/val"""
        if self.dataset_name == DatasetType.AG_NEWS:
            dataset = load_dataset("ag_news", split=["train", "test"])
            return dataset[0], dataset[1]
        elif self.dataset_name == DatasetType.OPENWEBTEXT:
            dataset = load_dataset("openwebtext", split="train")
            split_dataset = dataset.train_test_split(test_size=0.2)
            return split_dataset["train"], split_dataset["test"]
        elif self.dataset_name == DatasetType.WIKITEXT:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            return dataset["train"], dataset["validation"]
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def build_vocab(self):
        self.logger.info("Using pretrained tokenizer vocab...")
        return self.tokenizer.vocab

    def setup_data(self, device_for_generator: str = "cpu"):
        """
        Build PyTorch DataLoaders for training and validation.
        Args:
            device_for_generator: Device on which to create the PyTorch generator for shuffling.
                                Default is 'cpu'.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        train_data, val_data = self.download_dataset()
        vocab = self.build_vocab()
        train_dataset = TextDataset(
            train_data, tokenizer=self.tokenizer, seq_length=self.seq_len
        )
        val_dataset = TextDataset(
            val_data, tokenizer=self.tokenizer, seq_length=self.seq_len
        )

        train_loader = DattabotDataLoader(
            dataset=train_dataset,
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator(device=device_for_generator).manual_seed(42),
        )
        val_loader = DattabotDataLoader(
            dataset=val_dataset,
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return train_loader, val_loader, vocab
