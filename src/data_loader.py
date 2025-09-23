import os
from enum import Enum
from typing import Iterator, Optional, Union

import torch
import torch.distributed as dist
from datasets import load_dataset

from src.agent_config import get_agent_config
from src.logger import get_logger
from src.tokenizer import DattaBotTokenizer, get_tokenizer
from src.util import get_logging_level_from_config
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class DatasetType(Enum):
    OPENWEBTEXT = "openwebtext"
    AG_NEWS = "ag_news"
    WIKITEXT = "wikitext"
    FINANCEQA = "financeqa"
    MMLU_REDUX = "mmlu_redux"


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
        self, data: list[dict], tokenizer: DattaBotTokenizer, seq_length: int = 256
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        item = self.data[idx]

        if isinstance(item, dict):
            # For FinQA: concatenate question + context
            if "question" in item and "context" in item:
                text = f"Question: {item['question']} Context: {item['context']}"
            elif "text" in item:
                text = item["text"]
            else:
                # fallback: stringify dict
                text = str(item)
        else:
            text = item

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
    """Data builder supporting FinanceQA, MMLU-Redux, and standard datasets."""

    def __init__(self):
        self.config = get_agent_config()
        self.logger = get_logger(
            logging_level=get_logging_level_from_config(self.config)
        )
        self.tokenizer = get_tokenizer()
        self.batch_size = self.config.agent.batch_size
        self.seq_len = self.config.agent.max_response_tokens
        self.data_dir = self.config.agent.data_directory
        self.dataset_name = string_to_enum(self.config.env.dataset_name)

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

        elif self.dataset_name == DatasetType.FINANCEQA:
            dataset = load_dataset("ibm-research/finqa", split="train")
            split_dataset = dataset.train_test_split(test_size=0.1)
            return split_dataset["train"], split_dataset["test"]

        elif self.dataset_name == DatasetType.MMLU_REDUX:
            dataset = load_dataset("mmlu-redux", split="train")
            split_dataset = dataset.train_test_split(test_size=0.1)
            return split_dataset["train"], split_dataset["test"]

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def build_vocab(self):
        self.logger.info("Using pretrained tokenizer vocab...")
        return self.tokenizer.vocab

    def get_distributed_sampler(self, dataset, shuffle: bool):
        """Return a DistributedSampler if in DDP, otherwise None."""
        if dist.is_available() and dist.is_initialized():
            return DistributedSampler(dataset, shuffle=shuffle)
        return None

    def setup_data(self):
        """Build PyTorch DataLoaders for training and validation."""
        os.makedirs(self.data_dir, exist_ok=True)
        train_data, val_data = self.download_dataset()
        vocab = self.build_vocab()
        train_dataset = TextDataset(
            train_data, tokenizer=self.tokenizer, seq_length=self.seq_len
        )
        val_dataset = TextDataset(
            val_data, tokenizer=self.tokenizer, seq_length=self.seq_len
        )
        # Distributed samplers
        generator: torch.Generator = torch.Generator().manual_seed(42)
        train_sampler = self.get_distributed_sampler(train_dataset, shuffle=True)
        val_sampler = self.get_distributed_sampler(val_dataset, shuffle=False)

        train_loader = DattabotDataLoader(
            dataset=train_dataset,
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            sampler=train_sampler,
            # Only shuffle if not distributed
            shuffle=(train_sampler is None),
            generator=generator,
        )
        val_loader = DattabotDataLoader(
            dataset=val_dataset,
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            sampler=val_sampler,
            shuffle=False,
        )

        return train_loader, val_loader, vocab
