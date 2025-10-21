import math
from unittest.mock import Mock, patch

import pytest
import torch
from src.api_interface import DattaBotAPIResponse
from src.util import is_device_cpu


def test_is_device_cpu():
    assert is_device_cpu("cpu") == True
    assert is_device_cpu("cuda") == False
    assert is_device_cpu("cuda:0") == False
    assert is_device_cpu("cuda:1") == False
    assert is_device_cpu("cuda:234234") == False
    assert is_device_cpu("cuda:0,1") == False


@pytest.fixture
def mock_agent():
    """Create a mocked agent for fast unit tests."""
    with patch("src.agent.Agent") as MockAgent:
        from torch import device as torch_device

        agent = MockAgent.return_value

        # Mock tokenizer
        agent.tokenizer = Mock()
        agent.tokenizer.encode.return_value = [[1, 2, 3]]
        agent.tokenizer.decode.return_value = ["Hello! How can I help?"]

        # Mock model
        agent.model = Mock()

        # Mock comm_manager
        agent.comm_manager = Mock()
        agent.comm_manager.get_history.return_value = [
            {"role": "user", "content": "Hello DattaBot!"},
            {"role": "agent", "content": "Hello! How can I help?"},
        ]

        # Mock attributes
        agent.batch_size = 4
        agent.device = torch_device("cpu")

        # Mock convert_queries_to_tensors
        def mock_convert(queries):
            batch_size = len(queries)
            seq_len = 5
            tensor = torch.randint(0, 1000, (batch_size, seq_len))
            num_batches = math.ceil(batch_size / agent.batch_size)
            return tensor, num_batches

        agent.convert_queries_to_tensors = Mock(side_effect=mock_convert)

        # Mock respond_to_queries
        def mock_respond(queries):
            responses = []
            for i, query in enumerate(queries):
                response = DattaBotAPIResponse(
                    response_dict={
                        "output_text": f"Mock response to: {query}",
                        "choices": [
                            {
                                "text": f"Mock response to: {query}",
                                "index": 0,
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 10,
                            "total_tokens": 15,
                        },
                    },
                    metadata={
                        "tensor_response": torch.randint(0, 1000, (15,)),
                        "tokenizer_encodings": [[1, 2, 3, 4, 5]],
                        "tokenizer_decodings": [f"Mock response to: {query}"],
                        "inference_time": 0.01,
                        "tokens_per_second": 1500.0,
                    },
                )
                responses.append(response)
            return responses

        agent.respond_to_queries = Mock(side_effect=mock_respond)

        return agent


def test_agent_initialization(mock_agent):
    """Test agent initializes correctly."""
    assert mock_agent.tokenizer is not None
    assert mock_agent.model is not None
    assert mock_agent.comm_manager is not None
    assert isinstance(mock_agent.batch_size, int)
    assert mock_agent.batch_size > 0


def test_convert_queries_to_tensors(mock_agent):
    """Test query to tensor conversion."""
    queries = ["Hello world", "Another test query"]
    tensor, num_batches = mock_agent.convert_queries_to_tensors(queries)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 2  # (batch, seq_len)
    assert num_batches == math.ceil(len(queries) / mock_agent.batch_size)
    assert tensor.size(0) == len(queries)


def test_inference_and_response(mock_agent):
    """Test inference and response structure."""
    queries = ["Hello DattaBot!"]
    responses = mock_agent.respond_to_queries(queries)

    # Check response structure
    assert isinstance(responses, list)
    assert len(responses) == 1
    assert all(isinstance(r, DattaBotAPIResponse) for r in responses)

    r = responses[0]
    assert isinstance(r.text, str)
    assert isinstance(r.tensor_response, torch.Tensor)
    assert isinstance(r.tokenizer_encodings, list)
    assert isinstance(r.tokenizer_decodings, list)

    # Communication manager should have tracked history
    history = mock_agent.comm_manager.get_history()
    assert len(history) > 0
    assert any(msg["role"] == "user" for msg in history)
    assert any(msg["role"] == "agent" for msg in history)


def test_respond_to_queries(mock_agent):
    """Test multiple query responses."""
    queries = ["Test me please!", "Another query"]
    responses = mock_agent.respond_to_queries(queries)

    assert isinstance(responses, list)
    assert all(isinstance(r, DattaBotAPIResponse) for r in responses)
    assert len(responses) == len(queries)

    # The first response should be a string
    assert isinstance(responses[0].text, str)


# ============================================================
# INTEGRATION TESTS (slow, run with --integration flag)
# ============================================================


@pytest.fixture
def real_agent():
    """Real agent for integration tests - SLOW!"""
    from src.agent import Agent

    return Agent()


@pytest.mark.integration
def test_real_inference_and_response(real_agent):
    """Integration test with real inference - SLOW!"""
    queries = ["Hello DattaBot!"]
    responses = real_agent.respond_to_queries(queries)

    assert isinstance(responses, list)
    assert len(responses) == 1
    assert all(isinstance(r, DattaBotAPIResponse) for r in responses)

    r = responses[0]
    assert isinstance(r.text, str)
    assert isinstance(r.tensor_response, torch.Tensor)


@pytest.mark.integration
def test_real_respond_to_queries(real_agent):
    """Integration test with real responses - SLOW!"""
    queries = ["Test me please!"]
    responses = real_agent.respond_to_queries(queries)

    assert isinstance(responses, list)
    assert len(responses) == len(queries)
    assert isinstance(responses[0].text, str)
