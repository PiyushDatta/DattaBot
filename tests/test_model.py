import pytest
import torch
import torch.cuda.amp as amp
from src.model import DattaBotModel
from src.tokenizer import get_tokenizer
from src.util import is_device_cpu


@pytest.mark.parametrize("batch_size,seq_len", [(2, 4), (3, 8)])
def test_decoder_only_forward(batch_size, seq_len):
    model = DattaBotModel()
    model.eval()
    tokenizer = get_tokenizer()
    d_model = model.d_model
    # Fake batch of token IDs in vocab range.
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=model.device)
    # Fake labels (targets) with the same shape and valid vocab range.
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=model.device)

    # Check input and label type & shape
    # input = batch_size, seq_len
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert input_ids.shape == (batch_size, seq_len)
    assert labels.shape == (batch_size, seq_len)

    # Foward pass.
    with amp.autocast(
        enabled=True,
        dtype=torch.bfloat16,
    ):
        logits = model(input_ids)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (batch_size, seq_len, d_model)
        loss = logits.mean()

    # Backward pass.
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
