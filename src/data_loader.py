import os
from enum import Enum
from typing import Optional, Iterator
from src.agent_config import get_agent_config
from src.data_formatter import DattaBotDataFormatter
from src.logger import get_logger
from src.tokenizer import DattaBotTokenizer, get_tokenizer
from src.util import DatasetType, get_logging_level_from_config


def string_to_enum(dataset_name: str) -> DatasetType:
    try:
        return DatasetType[dataset_name.upper()]
    except KeyError:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. Options: {list(DatasetType._member_names_)}"
        )


# Defer the actual class creation until torch is imported
_TextDataset = None


def _create_text_dataset_class():
    """Create TextDataset class with proper Dataset inheritance only when needed"""
    global _TextDataset
    if _TextDataset is not None:
        return _TextDataset

    from torch.utils.data import Dataset
    import torch

    class TextDataset(Dataset):
        """Custom dataset for GPT-style training"""

        def __init__(
            self,
            data: list[dict],
            dataset_type: str,
            tokenizer: DattaBotTokenizer,
            seq_length: int = 256,
        ):
            super().__init__()
            get_logger().info(
                f"Setting up TextDataset with dataset type {dataset_type} with sequence length: {seq_length}"
            )
            self.data = data
            self.tokenizer = tokenizer
            self.seq_length = seq_length
            self.formatter = DattaBotDataFormatter(dataset_type=dataset_type)

        def __len__(self) -> int:
            return len(self.data)

        def get_raw_formatted_text(self, idx) -> str:
            """Get formatted text using the dataset-specific formatter"""
            item = self.data[idx]
            return self.formatter.format(item=item, max_seq_len=self.seq_length)

        def __getitem__(self, idx: int):
            raw_text = self.get_raw_formatted_text(idx=idx)
            # Encode the text (BOS/EOS automatically added by the tokenizer)
            tokens = self.tokenizer.encode(raw_text)
            # Ensure tokens are of length seq_length + 1 for x/y splitting
            if len(tokens) > self.seq_length + 1:
                # Take first seq_length-1 tokens, append EOS, then pad or adjust
                tokens = tokens[: self.seq_length - 1] + [self.tokenizer.eos_token_id]
                # Append one more token (e.g., pad or next token) to reach seq_length + 1
                tokens.append(
                    self.tokenizer.pad_token_id
                    if len(tokens) < self.seq_length + 1
                    else tokens[self.seq_length - 1]
                )
            elif len(tokens) < self.seq_length + 1:
                # Pad to seq_length + 1
                tokens += [self.tokenizer.pad_token_id] * (
                    (self.seq_length + 1) - len(tokens)
                )
            else:
                # If exactly seq_length + 1, ensure EOS is at seq_length-1
                tokens[self.seq_length - 1] = self.tokenizer.eos_token_id
            # Verify length
            assert (
                len(tokens) == self.seq_length + 1
            ), f"Tokens length {len(tokens)} != {self.seq_length + 1}"
            # Prepare x and y for next-token prediction
            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long)
            assert (
                len(x) == self.seq_length and len(y) == self.seq_length
            ), f"x length {len(x)} or y length {len(y)} != seq_length {self.seq_length}"
            return x, y, raw_text

    _TextDataset = TextDataset
    return _TextDataset


class DattabotDataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        reset_batch_when_reach_end: bool = True,
        **kwargs,
    ):
        from torch.utils.data import DataLoader as TorchDataLoader

        self._reset_batch_end = reset_batch_when_reach_end
        self._dataset = dataset
        self._iterator: Optional[Iterator] = None
        self._dataloader = TorchDataLoader(dataset, batch_size=batch_size, **kwargs)

    def __iter__(self) -> Iterator:
        self._iterator = iter(self._dataloader)
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self._dataloader)
        try:
            return next(self._iterator)
        except StopIteration:
            if self._reset_batch_end:
                self._iterator = iter(self._dataloader)
                return next(self._iterator)
            else:
                raise

    def __len__(self):
        return len(self._dataloader)


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
        self.dataset_type = string_to_enum(self.config.env.dataset_name)

    def download_dataset(self):
        """Download dataset and split into train/val"""
        from datasets import load_dataset

        if self.dataset_type == DatasetType.AG_NEWS:
            dataset = load_dataset("ag_news", split=["train", "test"])
            return dataset[0], dataset[1]

        elif self.dataset_type == DatasetType.OPENWEBTEXT:
            dataset = load_dataset("openwebtext", split="train")
            split_dataset = dataset.train_test_split(test_size=0.2)
            return split_dataset["train"], split_dataset["test"]

        elif self.dataset_type == DatasetType.WIKITEXT:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            return dataset["train"], dataset["validation"]

        elif self.dataset_type == DatasetType.FINANCEQA:
            dataset = load_dataset("ibm-research/finqa", split="train")
            split_dataset = dataset.train_test_split(test_size=0.1)
            return split_dataset["train"], split_dataset["test"]

        elif self.dataset_type == DatasetType.MMLU_REDUX:
            dataset = load_dataset("mmlu-redux", split="train")
            split_dataset = dataset.train_test_split(test_size=0.1)
            return split_dataset["train"], split_dataset["test"]

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_type}")

    def build_vocab(self):
        self.logger.info("Using pretrained tokenizer vocab...")
        return self.tokenizer.vocab

    def get_distributed_sampler(self, dataset, shuffle: bool):
        """Return a DistributedSampler if in DDP, otherwise None."""
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler

        if dist.is_available() and dist.is_initialized():
            return DistributedSampler(dataset, shuffle=shuffle)
        return None

    def setup_data(self):
        """Build PyTorch DataLoaders for training and validation."""
        self.logger.info("Setting up data..")
        import torch

        # Get the properly defined TextDataset class
        TextDataset = _create_text_dataset_class()

        os.makedirs(self.data_dir, exist_ok=True)
        train_data, val_data = self.download_dataset()
        vocab = self.build_vocab()
        train_dataset = TextDataset(
            data=train_data,
            dataset_type=self.dataset_type,
            tokenizer=self.tokenizer,
            seq_length=self.seq_len,
        )
        val_dataset = TextDataset(
            data=val_data,
            dataset_type=self.dataset_type,
            tokenizer=self.tokenizer,
            seq_length=self.seq_len,
        )
        # Distributed samplers
        generator: torch.Generator = torch.Generator().manual_seed(42)
        train_sampler = self.get_distributed_sampler(train_dataset, shuffle=True)
        val_sampler = self.get_distributed_sampler(val_dataset, shuffle=False)

        train_loader = DattabotDataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            # Only shuffle if not distributed
            shuffle=(train_sampler is None),
            generator=generator,
        )
        val_loader = DattabotDataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            shuffle=False,
        )
        self.logger.info("Finished loading data.")
        return train_loader, val_loader, vocab

    def get_random_example(self) -> tuple[dict, dict]:
        """
        Returns a single random (x, y) pair from the validation/test dataloader.
        Mimics exactly what is yielded by next(val_dataloader).
        """
        import torch

        # Build validation dataloader exactly like in the training setup.
        _, val_loader, _ = self.setup_data()
        # Pick a random batch from the val loader. Cap at max_batches to avoid iterating too much.
        self.logger.info("Picking random batch from the validation loader...")
        max_batches = 30
        random_batch_index = torch.randint(
            0, min(max_batches, len(val_loader)), (1,)
        ).item()
        self.logger.info(f"Random batch idx: {random_batch_index}")
        batch_iter = iter(val_loader)
        # Advance the iterator to the chosen batch
        for _ in range(random_batch_index):
            next(batch_iter)
        # x, y, raw_text_batch = next(batch_iter)
        batch = next(batch_iter)
        batch_len = len(batch)
        self.logger.info(
            f"Got batch from random batch idx({random_batch_index}), len of batch: {batch_len}"
        )
        if batch_len == 3:
            x, y, raw_text_batch = batch
        else:
            # backward compatible if older dataset version
            x, y = batch
            raw_text_batch = ["<unknown>"] * x.size(0)

        # Choose a random example within that batch
        rand_idx = torch.randint(0, x.size(0), (1,)).item()
        x_item = x[rand_idx]
        y_item = y[rand_idx]
        raw_text = (
            raw_text_batch[rand_idx]
            if isinstance(raw_text_batch, list) or isinstance(raw_text_batch, tuple)
            else raw_text_batch
        )

        decoded_input = self.tokenizer.decode(x_item.tolist())
        decoded_target = self.tokenizer.decode(y_item.tolist())
        # Build metadata
        metadata = {
            "raw_text": raw_text,
            "seq_len": len(x_item),
            "tensor_response": {"x": x_item, "y": y_item},
            "tokenizer_encodings": [x_item.tolist(), y_item.tolist()],
            "tokenizer_decodings": [decoded_input, decoded_target],
        }
        # Build raw response dict
        response_dict = {
            "output_text": decoded_target,
            "choices": [{"input_text": decoded_input, "target_text": decoded_target}],
        }
        return response_dict, metadata
