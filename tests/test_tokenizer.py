import pytest
from src.tokenizer import get_tokenizer


@pytest.fixture
def tokenizer():
    return get_tokenizer()


def test_singleton_behavior():
    t1 = get_tokenizer()
    t2 = get_tokenizer()
    assert t1 is t2, "Tokenizer should be a singleton"


def test_encode_single_string(tokenizer):
    text = "Hello world!"
    encoded = tokenizer.encode(text)
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id


def test_encode_batch_strings(tokenizer):
    texts = ["Hello", "world!"]
    encoded_batch = tokenizer.encode(texts)
    assert isinstance(encoded_batch, list)
    assert len(encoded_batch) == len(texts)
    for seq in encoded_batch:
        assert seq[0] == tokenizer.bos_token_id
        assert seq[-1] == tokenizer.eos_token_id


def test_decode_single_sequence(tokenizer):
    text = "Test decoding"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)
    assert text in decoded  # decoded may include extra tokens


def test_decode_batch_sequences(tokenizer):
    texts = ["Hello", "world!"]
    encoded_batch = tokenizer.encode(texts)
    decoded_batch = tokenizer.decode(encoded_batch)
    assert isinstance(decoded_batch, list)
    assert len(decoded_batch) == len(texts)
    for original, decoded in zip(texts, decoded_batch):
        assert original in decoded


def test_token_to_id_and_id_to_token(tokenizer):
    special = tokenizer.bos_token
    token_id = tokenizer.token_to_id(special)
    assert token_id == tokenizer.bos_token_id
    token_str = tokenizer.id_to_token(token_id)
    assert token_str == special


def test_vocab_property(tokenizer):
    vocab = tokenizer.vocab
    assert isinstance(vocab, dict)
    assert len(vocab) == tokenizer.vocab_size
    # Check that special tokens exist in vocab
    assert tokenizer.bos_token in vocab
    assert tokenizer.eos_token in vocab
    assert vocab[tokenizer.bos_token] == tokenizer.bos_token_id
    assert vocab[tokenizer.eos_token] == tokenizer.eos_token_id


def test_vocab_size_property(tokenizer):
    assert isinstance(tokenizer.vocab_size, int)
    assert tokenizer.vocab_size > 0


def test_special_tokens(tokenizer):
    assert tokenizer.bos_token_id != -1
    assert tokenizer.eos_token_id != -1
    assert tokenizer.pad_token_id != -1


def test_invalid_encode_input(tokenizer):
    with pytest.raises(TypeError):
        tokenizer.encode(123)  # not str or list[str]


def test_invalid_decode_input(tokenizer):
    with pytest.raises(TypeError):
        tokenizer.decode(123)  # not list[int] or list[list[int]]
