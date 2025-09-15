import torch
import pytest
from src.agent import Agent
from src.api_interface import DattaBotAPIResponse


@pytest.fixture
def agent():
    # Initialize agent once per test session
    return Agent()


def test_agent_initialization(agent):
    # Make sure core components exist
    assert agent.tokenizer is not None
    assert agent.model is not None
    assert agent.comm_manager is not None
    assert isinstance(agent.batch_size, int)
    assert agent.batch_size > 0


def test_convert_queries_to_tensors(agent):
    queries = ["Hello world", "Another test query"]
    tensor, num_batches = agent.convert_queries_to_tensors(queries)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 2  # (batch, seq_len)
    assert num_batches == 1
    assert tensor.size(0) == len(queries)


def test_inference_and_response(agent):
    queries = ["Hello DattaBot!"]
    responses = agent._inference(queries)

    # Check response structure
    assert isinstance(responses, list)
    assert len(responses) == 1
    assert all(isinstance(r, DattaBotAPIResponse) for r in responses)

    r = responses[0]
    assert isinstance(r.text, str)
    assert isinstance(r.tensor_response, torch.Tensor)
    assert isinstance(r.tokenizer_encodings, list)
    assert isinstance(r.tokenizer_decodings, str)

    # Communication manager should have tracked history
    history = agent.comm_manager.get_history()
    assert any(msg["role"] == "user" for msg in history)
    assert any(msg["role"] == "agent" for msg in history)


def test_respond_to_queries(agent):
    queries = ["Test me please!"]
    responses: list[DattaBotAPIResponse] = agent.respond_to_queries(queries)

    assert isinstance(responses, list)
    assert all(isinstance(r, DattaBotAPIResponse) for r in responses)
    assert len(responses) == len(queries)

    # The first response should be a string
    assert isinstance(responses[0].text, str)
