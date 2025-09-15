import torch
import pytest
from src.model import DattaBotModel
from src.tokenizer import get_tokenizer


@pytest.mark.parametrize("batch_size,seq_len", [(2, 4), (3, 8)])
def test_decoder_only_forward(batch_size, seq_len):
    model = DattaBotModel()
    tokenizer = get_tokenizer()

    # Fake batch of token IDs in vocab range
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=model.device)

    logits = model(input_ids)

    # Check output type & shape
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, seq_len, vocab_size)

    # Backward pass should work
    loss = logits.mean()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
