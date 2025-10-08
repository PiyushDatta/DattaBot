import sys

import pytest
import torch.distributed as dist
from torch import Tensor

# set path to project
sys.path.append("../dattabot")

from src.agent_config import get_agent_config
from src.api import DattaBotAPI, DattaBotAPIException, DattaBotAPIResponse
from src.logger import get_logger
from src.util import setup_torch_dist_init

setup_torch_dist_init()
logger = get_logger()
config = get_agent_config()
datta_bot_api = DattaBotAPI()


def test_smoke_dattabot():
    """Simple smoke test"""
    assert 1 == 1


def test_tensor_encoding_one_query():
    one_query = ["Helloooo!"]
    responses: list[DattaBotAPIResponse] = datta_bot_api.get_tensor_encoding(one_query)
    resp = responses[0]

    assert resp is not None
    assert hasattr(resp, "tensor_response")
    assert isinstance(resp.tensor_response, Tensor)
    # batch dimension
    assert resp.tensor_response.size(0) == len(one_query), (
        f"Expected batch size {len(one_query)}, "
        f"but got {resp.tensor_response.size(0)}"
    )
    # (batch, seq_len)
    assert resp.tensor_response.ndim == 2, (
        f"Expected tensor to have 2 dimensions (batch, seq_len), "
        f"but got {resp.tensor_response.ndim}"
    )


def test_tensor_encoding_two_queries():
    two_queries = ["Hello!", "We've met already."]
    responses: list[DattaBotAPIResponse] = datta_bot_api.get_tensor_encoding(
        two_queries
    )
    for resp in responses:
        assert resp is not None
        assert hasattr(resp, "tensor_response")
        assert isinstance(resp.tensor_response, Tensor)
        # batch dimension
        assert resp.tensor_response.size(0) == len(two_queries), (
            f"Expected batch size {len(two_queries)}, "
            f"but got {resp.tensor_response.size(0)}"
        )
        # (batch, seq_len)
        assert resp.tensor_response.ndim == 2, (
            f"Expected tensor to have 2 dimensions (batch, seq_len), "
            f"but got {resp.tensor_response.ndim}"
        )


@pytest.mark.integration
def test_real_single_response_text():
    query = "How are you?"
    resp: DattaBotAPIResponse = datta_bot_api.get_response(query)
    assert isinstance(resp, DattaBotAPIResponse)
    assert hasattr(resp, "text")
    assert isinstance(resp.text, str)


def test_encoding_and_decoding_roundtrip():
    queries = ["Roundtrip test."]
    encoding_resp: DattaBotAPIResponse = datta_bot_api.get_encoding(queries)[0]
    assert hasattr(encoding_resp, "tokenizer_encodings")
    encodings = encoding_resp.tokenizer_encodings
    assert isinstance(encodings, list)
    assert all(isinstance(seq, list) for seq in encodings)

    decoding_resp: DattaBotAPIResponse = datta_bot_api.get_decoding(encodings)[0]
    assert hasattr(decoding_resp, "tokenizer_decodings")
    decodings = decoding_resp.tokenizer_decodings
    assert isinstance(decodings, list)
    assert all(isinstance(text, str) for text in decodings)
