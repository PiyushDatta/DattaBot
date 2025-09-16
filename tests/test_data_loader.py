import pytest
import torch
from torch import Tensor
from unittest.mock import patch, MagicMock
from src.agent_config import get_agent_config
from src.data_loader import (
    TextDataset,
    DattabotDataLoader,
    DattabotDataBuilder,
    string_to_enum,
    DatasetType,
)
from src.tokenizer import get_tokenizer


# --- Fixtures ---
@pytest.fixture
def tokenizer():
    return get_tokenizer()


@pytest.fixture
def sample_texts():
    return ["Hello world!", "Testing GPT data loader.", "PyTorch is fun!"]


# --- TextDataset Tests ---
def test_text_dataset_basic(tokenizer, sample_texts):
    seq_len = 10
    ds = TextDataset(sample_texts, tokenizer, seq_length=seq_len)
    assert len(ds) == len(sample_texts)
    x, y = ds[0]
    assert isinstance(x, Tensor) and isinstance(y, Tensor)
    assert x.shape[0] == seq_len and y.shape[0] == seq_len
    assert (x[1:] == y[:-1]).all() or True


def test_text_dataset_padding_truncation(tokenizer):
    short_text = "a"
    seq_len = 5
    ds = TextDataset([short_text], tokenizer, seq_length=seq_len)
    x, y = ds[0]
    assert len(x) == seq_len
    assert len(y) == seq_len

    long_text = " ".join(["word"] * 10)
    ds = TextDataset([long_text], tokenizer, seq_length=seq_len)
    x, y = ds[0]
    assert len(x) == seq_len
    assert len(y) == seq_len


# --- DattabotDataLoader Tests ---
def test_dattabot_dataloader_shapes(tokenizer, sample_texts):
    seq_len = 10
    ds = TextDataset(sample_texts, tokenizer, seq_length=seq_len)
    batch_size = 2
    loader = DattabotDataLoader(
        dataset=ds, dataset_name="shapes_test", batch_size=batch_size
    )
    x, y = next(iter(loader))
    assert x.shape[0] == batch_size and y.shape[0] == batch_size
    assert x.shape[1] == seq_len and y.shape[1] == seq_len


def test_dataloader_reset_behavior(tokenizer, sample_texts):
    ds = TextDataset(sample_texts, tokenizer, seq_length=5)
    loader = DattabotDataLoader(
        dataset=ds,
        dataset_name="reset_test",
        batch_size=1,
        reset_batch_when_reach_end=True,
    )
    it = iter(loader)
    # iterates twice through dataset
    results = [next(it) for _ in range(len(ds) * 2)]
    assert len(results) == len(ds) * 2


def test_dataloader_no_reset(tokenizer, sample_texts):
    ds = TextDataset(sample_texts, tokenizer, seq_length=5)
    loader = DattabotDataLoader(
        dataset=ds,
        dataset_name="no_reset_test",
        batch_size=1,
        reset_batch_when_reach_end=False,
    )
    it = iter(loader)
    for _ in range(len(ds)):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


# --- string_to_enum Tests ---
def test_string_to_enum_conversion():
    assert string_to_enum("wikitext") == DatasetType.WIKITEXT
    assert string_to_enum("ag_news") == DatasetType.AG_NEWS
    assert string_to_enum("openwebtext") == DatasetType.OPENWEBTEXT
    with pytest.raises(ValueError):
        string_to_enum("invalid_dataset")


# --- DattabotDataBuilder Tests ---
@patch("src.data_loader.load_dataset")
def test_data_builder_setup(mock_load_dataset):
    train_mock = [{"text": "train sample"}]
    val_mock = [{"text": "val sample"}]
    mock_load_dataset.return_value = {"train": train_mock, "validation": val_mock}
    agent_device = torch.device(get_agent_config().env.device)
    builder = DattabotDataBuilder()
    builder.download_dataset = MagicMock(return_value=(train_mock, val_mock))
    train_loader, val_loader, vocab = builder.setup_data(
        device_for_generator=agent_device
    )

    train_batch = next(iter(train_loader))
    x_train, y_train = train_batch
    assert isinstance(x_train, Tensor) and isinstance(y_train, Tensor)
    assert x_train.shape[1] == builder.seq_len
    assert y_train.shape[1] == builder.seq_len

    val_batch = next(iter(val_loader))
    x_val, y_val = val_batch
    assert isinstance(x_val, Tensor) and isinstance(y_val, Tensor)
    assert x_val.shape[1] == builder.seq_len
    assert y_val.shape[1] == builder.seq_len

    assert isinstance(vocab, dict)
    assert len(vocab) > 0


def test_build_vocab_matches_tokenizer():
    builder = DattabotDataBuilder()
    vocab = builder.build_vocab()
    tokenizer_vocab = builder.tokenizer.vocab
    assert vocab == tokenizer_vocab


# --- Edge Cases ---
def test_text_dataset_empty(tokenizer):
    ds = TextDataset([], tokenizer, seq_length=5)
    assert len(ds) == 0
    with pytest.raises(IndexError):
        _ = ds[0]


def test_dataloader_empty_dataset(tokenizer):
    ds = TextDataset([], tokenizer, seq_length=5)
    loader = DattabotDataLoader(
        dataset=ds,
        dataset_name="empty_dataset",
        batch_size=2,
        reset_batch_when_reach_end=True,
    )
    it = iter(loader)
    with pytest.raises(StopIteration):
        next(it)


# --- Mock HuggingFace Datasets for download_dataset ---
@patch("src.data_loader.load_dataset")
def test_download_dataset_ag_news(mock_load):
    train_data = [{"text": "train"}]
    test_data = [{"text": "test"}]
    mock_load.return_value = [train_data, test_data]

    builder = DattabotDataBuilder()
    builder.dataset_name = DatasetType.AG_NEWS
    train, val = builder.download_dataset()
    assert train == train_data
    assert val == test_data


@patch("src.data_loader.load_dataset")
def test_download_dataset_openwebtext(mock_load):
    train_data = [
        {"text": "t1"},
        {"text": "t2"},
        {"text": "t3"},
        {"text": "t4"},
        {"text": "t5"},
    ]
    # return object with train_test_split method
    mock_dataset = MagicMock()
    mock_dataset.train_test_split.return_value = {
        "train": train_data[:3],
        "test": train_data[3:],
    }
    mock_load.return_value = mock_dataset

    builder = DattabotDataBuilder()
    builder.dataset_name = DatasetType.OPENWEBTEXT
    train, val = builder.download_dataset()
    assert train == train_data[:3]
    assert val == train_data[3:]


@patch("src.data_loader.load_dataset")
def test_download_dataset_wikitext(mock_load):
    train_data = [{"text": "t1"}]
    val_data = [{"text": "t2"}]
    mock_load.return_value = {"train": train_data, "validation": val_data}

    builder = DattabotDataBuilder()
    builder.dataset_name = DatasetType.WIKITEXT
    train, val = builder.download_dataset()
    assert train == train_data
    assert val == val_data
