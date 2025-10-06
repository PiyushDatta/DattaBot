from unittest.mock import MagicMock, patch
import pytest
from src.data_loader import (
    DattabotDataBuilder,
    DattabotDataLoader,
    string_to_enum,
    _create_text_dataset_class,
)
from src.util import DatasetType
from torch import Tensor


# --- Fixtures ---
@pytest.fixture
def tokenizer():
    # Mock tokenizer that returns token ids for strings
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode = lambda text: [ord(c) for c in str(text)]
    mock_tokenizer.decode = lambda tokens: "".join(chr(t) for t in tokens)
    mock_tokenizer.vocab = {str(i): i for i in range(256)}
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.pad_token_id = 1
    return mock_tokenizer


@pytest.fixture
def sample_texts():
    # Format as list of dicts to match data_loader.py expectation
    return [
        {"text": "Hello world!"},
        {"text": "Testing GPT data loader."},
        {"text": "PyTorch is fun!"},
    ]


# --- TextDataset Tests ---
def test_text_dataset_basic(tokenizer, sample_texts):
    seq_length = 10
    TextDataset = _create_text_dataset_class()
    ds = TextDataset(
        data=sample_texts,
        dataset_type="wikitext",
        tokenizer=tokenizer,
        seq_length=seq_length,
    )
    x, y, raw_text = ds[0]
    assert isinstance(x, Tensor) and isinstance(y, Tensor)
    assert x.shape[0] == seq_length and y.shape[0] == seq_length
    assert isinstance(raw_text, str)


def test_text_dataset_padding_truncation(tokenizer):
    seq_length = 5
    TextDataset = _create_text_dataset_class()

    # Test padding (short text)
    ds = TextDataset(
        data=[{"text": "a"}],
        dataset_type="wikitext",
        tokenizer=tokenizer,
        seq_length=seq_length,
    )
    x, y, _ = ds[0]
    assert len(x) == seq_length and len(y) == seq_length
    # Check padding
    assert x[-1] == tokenizer.pad_token_id

    # Test truncation (long text)
    ds = TextDataset(
        data=[{"text": "word " * 10}],
        dataset_type="wikitext",
        tokenizer=tokenizer,
        seq_length=seq_length,
    )
    x, y, _ = ds[0]
    assert len(x) == seq_length and len(y) == seq_length
    # Check truncation preserves EOS
    assert x[-1] == tokenizer.eos_token_id


# --- DattabotDataLoader Tests ---
def test_dattabot_dataloader_shapes(tokenizer, sample_texts):
    seq_length = 10
    TextDataset = _create_text_dataset_class()
    ds = TextDataset(
        data=sample_texts,
        dataset_type="wikitext",
        tokenizer=tokenizer,
        seq_length=seq_length,
    )
    loader = DattabotDataLoader(dataset=ds, batch_size=2)
    x, y, raw_text = next(iter(loader))
    assert x.shape[0] == 2 and y.shape[0] == 2
    assert x.shape[1] == seq_length and y.shape[1] == seq_length
    assert len(raw_text) == 2


def test_dataloader_reset_behavior(tokenizer, sample_texts):
    seq_length = 5
    TextDataset = _create_text_dataset_class()
    ds = TextDataset(
        data=sample_texts,
        dataset_type="wikitext",
        tokenizer=tokenizer,
        seq_length=seq_length,
    )
    loader = DattabotDataLoader(
        dataset=ds, batch_size=1, reset_batch_when_reach_end=True
    )
    it = iter(loader)
    results = [next(it) for _ in range(len(ds) * 2)]
    assert len(results) == len(ds) * 2
    assert all(len(batch) == 3 for batch in results)  # Check (x, y, raw_text)


def test_dataloader_no_reset(tokenizer, sample_texts):
    seq_length = 5
    TextDataset = _create_text_dataset_class()
    ds = TextDataset(
        data=sample_texts,
        dataset_type="wikitext",
        tokenizer=tokenizer,
        seq_length=seq_length,
    )
    loader = DattabotDataLoader(
        dataset=ds, batch_size=1, reset_batch_when_reach_end=False
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
@patch("datasets.load_dataset")
def test_data_builder_setup(mock_load_dataset, tokenizer):
    train_mock = [{"text": "train sample"}]
    val_mock = [{"text": "val sample"}]
    mock_load_dataset.return_value = {"train": train_mock, "validation": val_mock}

    builder = DattabotDataBuilder()
    builder.download_dataset = MagicMock(return_value=(train_mock, val_mock))
    builder.tokenizer = tokenizer
    builder.dataset_type = DatasetType.WIKITEXT
    train_loader, val_loader, vocab = builder.setup_data()

    x_train, y_train, raw_text_train = next(iter(train_loader))
    x_val, y_val, raw_text_val = next(iter(val_loader))

    for t in [x_train, y_train, x_val, y_val]:
        assert isinstance(t, Tensor)
        assert t.shape[1] == builder.seq_len

    assert isinstance(vocab, dict)
    assert len(vocab) > 0


def test_build_vocab_matches_tokenizer(tokenizer):
    builder = DattabotDataBuilder()
    builder.tokenizer = tokenizer
    vocab = builder.build_vocab()
    assert vocab == tokenizer.vocab


# --- Edge Cases ---
def test_text_dataset_empty(tokenizer):
    TextDataset = _create_text_dataset_class()
    ds = TextDataset(
        data=[],
        dataset_type="wikitext",
        tokenizer=tokenizer,
        seq_length=5,
    )
    assert len(ds) == 0
    with pytest.raises(IndexError):
        _ = ds[0]


def test_dataloader_empty_dataset(tokenizer):
    TextDataset = _create_text_dataset_class()
    ds = TextDataset(
        data=[],
        dataset_type="wikitext",
        tokenizer=tokenizer,
        seq_length=5,
    )
    loader = DattabotDataLoader(
        dataset=ds, batch_size=2, reset_batch_when_reach_end=True
    )
    it = iter(loader)
    with pytest.raises(StopIteration):
        next(it)


# --- Mock HuggingFace Datasets ---
@patch("datasets.load_dataset")
def test_download_dataset_ag_news(mock_load):
    train_data = [{"text": "train"}]
    test_data = [{"text": "test"}]
    mock_load.return_value = [train_data, test_data]

    builder = DattabotDataBuilder()
    builder.dataset_type = DatasetType.AG_NEWS
    train, val = builder.download_dataset()
    assert train == train_data
    assert val == test_data


@patch("datasets.load_dataset")
def test_download_dataset_openwebtext(mock_load):
    train_data = [{"text": f"t{i}"} for i in range(1, 6)]
    mock_dataset = MagicMock()
    mock_dataset.train_test_split.return_value = {
        "train": train_data[:3],
        "test": train_data[3:],
    }
    mock_load.return_value = mock_dataset

    builder = DattabotDataBuilder()
    builder.dataset_type = DatasetType.OPENWEBTEXT
    train, val = builder.download_dataset()
    assert train == train_data[:3]
    assert val == train_data[3:]


@patch("datasets.load_dataset")
def test_download_dataset_wikitext(mock_load):
    train_data = [{"text": "t1"}]
    val_data = [{"text": "t2"}]
    mock_load.return_value = {"train": train_data, "validation": val_data}

    builder = DattabotDataBuilder()
    builder.dataset_type = DatasetType.WIKITEXT
    train, val = builder.download_dataset()
    assert train == train_data
    assert val == val_data
