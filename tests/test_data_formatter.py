import pytest
from unittest.mock import patch, MagicMock
from src.data_formatter import DattaBotDataFormatter
from src.util import DatasetType


# ----------------------
# Fixtures
# ----------------------
@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, **kwargs: list(range(len(text.split())))
    tokenizer.decode.side_effect = lambda tokens, **kwargs: " ".join(
        f"T{t}" for t in tokens
    )
    return tokenizer


# ----------------------
# Patch get_tokenizer
# ----------------------
@pytest.fixture(autouse=True)
def patch_get_tokenizer(mock_tokenizer):
    with patch("src.data_formatter.get_tokenizer", return_value=mock_tokenizer):
        yield


# ----------------------
# Tests
# ----------------------
def test_format_financeqa_basic():
    item = {
        "question": "What is 2+2?",
        "answer": "4",
        "table": [[1, 2], [3, 4]],
        "pre_text": ["Pretext here."],
        "post_text": ["Posttext here."],
    }
    formatter = DattaBotDataFormatter(DatasetType.FINANCEQA)
    result = formatter.format(item)
    assert "Question: What is 2+2?" in result
    assert "Answer: 4" in result
    assert "Context:" in result
    assert "Table:" in result
    assert "Pretext" in result
    assert "Posttext" in result


def test_format_mmlu_redux_basic():
    item = {"question": "Q?", "choices": ["A", "B", "C"], "answer": "B"}
    formatter = DattaBotDataFormatter(DatasetType.MMLU_REDUX)
    result = formatter.format(item)
    assert "Question: Q?" in result
    assert "A. A" in result
    assert "B. B" in result
    assert "C. C" in result
    assert "Answer: B" in result


def test_format_ag_news_dict_and_str():
    dict_item = {"text": "Some news", "label": "World"}
    str_item = "Just a string news"
    formatter = DattaBotDataFormatter(DatasetType.AG_NEWS)

    result_dict = formatter.format(dict_item)
    assert "Some news" in result_dict
    assert "Category: World" in result_dict

    result_str = formatter.format(str_item)
    assert result_str == str_item


def test_format_wikitext_and_openwebtext():
    item_dict = {"text": "Wiki content"}
    item_str = "OpenWebText content"

    wtf = DattaBotDataFormatter(DatasetType.WIKITEXT)
    owf = DattaBotDataFormatter(DatasetType.OPENWEBTEXT)

    result_wiki = wtf.format(item_dict)
    result_ow = owf.format(item_str)

    assert "Wiki content" in result_wiki
    assert "OpenWebText content" in result_ow


def test_format_generic_dict_and_str():
    dict_item = {"text": "some text", "content": "fallback content"}
    str_item = "just string"
    formatter = DattaBotDataFormatter(
        DatasetType.FINANCEQA
    )  # dataset type won't use generic

    # directly call generic
    result_dict = formatter._format_generic(dict_item)
    result_str = formatter._format_generic(str_item)

    assert result_dict == "some text"  # picks "text" first
    assert result_str == str_item


def test_truncate_text_respects_max_len():
    formatter = DattaBotDataFormatter(DatasetType.FINANCEQA)
    long_text = " ".join(str(i) for i in range(50))
    truncated = formatter._truncate_text(long_text, max_seq_len=10)
    tokens = truncated.split()
    # Should be <= max_seq_len - 2
    assert len(tokens) <= 8


def test_repr_contains_dataset_type():
    formatter = DattaBotDataFormatter(DatasetType.AG_NEWS)
    expected_rep_grep = DatasetType.AG_NEWS.value
    rep = repr(formatter)
    assert (
        expected_rep_grep in rep
    ), f"Could not find text ({expected_rep_grep}) in rep. Found rep: {rep}."
