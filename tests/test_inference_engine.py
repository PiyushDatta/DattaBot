from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from src.api_interface import DattaBotAPIResponse

from src.inference_engine import DattaBotInferenceEngine
from src.model import DattaBotModel
from torch import nn


@pytest.fixture
def mock_config():
    """Mock agent configuration."""
    config = MagicMock()
    config.neural_net.n_layers = 2
    config.neural_net.n_heads = 4
    config.neural_net.model_dimensions = 128
    config.neural_net.hidden_layers = 512
    config.neural_net.zeroed_drop_probability = 0.1
    config.agent.max_response_tokens = 64
    config.agent.batch_size = 4
    config.env.device = "cpu"
    config.env.env_name = "test"
    config.env.logging_level = "INFO"

    # Add inference config
    config.inference.max_new_tokens = 20
    config.inference.temperature = 1.0
    config.inference.top_k = 50
    config.inference.top_p = 0.9
    config.inference.do_sample = False
    config.inference.num_beams = 1
    config.inference.repetition_penalty = 1.0
    config.inference.use_cache = True

    return config


@pytest.fixture
def mock_model(mock_config):
    """Create a mock model for testing."""
    with patch("src.model.get_agent_config", return_value=mock_config):
        with patch("src.model.get_tokenizer") as mock_tok:
            mock_tok.return_value.vocab_size = 1000
            model = DattaBotModel(device="cpu")
    return model


@pytest.fixture
def mock_adaptive_softmax(mock_config):
    """Create a REAL adaptive softmax (not a mock) for testing."""
    vocab_size = 1000
    d_model = mock_config.neural_net.model_dimensions
    # Create a REAL AdaptiveLogSoftmaxWithLoss, not a mock
    # This ensures all the weights are real tensors that support matrix operations
    adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
        in_features=d_model,
        n_classes=vocab_size,
        cutoffs=[500, vocab_size - 1],  # One tail cluster
        div_value=4.0,
        head_bias=True,
    )
    # The inference engine uses autocast with bfloat16, so adaptive_softmax must match
    adaptive_softmax = adaptive_softmax.to(dtype=torch.bfloat16)
    return adaptive_softmax


@pytest.fixture
def inference_engine(mock_model, mock_config, mock_adaptive_softmax):
    """Create inference engine."""
    with patch("src.inference_engine.get_agent_config", return_value=mock_config):
        with patch("src.inference_engine.get_logger"):
            with patch("src.inference_engine.get_tokenizer") as mock_tok:
                tokenizer = Mock()
                tokenizer.vocab_size = 1000
                tokenizer.pad_token_id = 0
                tokenizer.eos_token_id = 999
                tokenizer.encode = Mock(return_value=[1, 2, 3])

                def mock_decode(token_lists):
                    # token_lists is a list of token lists (one per batch item)
                    return [f"test output {i}" for i in range(len(token_lists))]

                tokenizer.decode = Mock(side_effect=mock_decode)
                mock_tok.return_value = tokenizer
                engine = DattaBotInferenceEngine(
                    model=mock_model,
                    adaptive_softmax=mock_adaptive_softmax,
                    device="cpu",
                )
    return engine


class TestDattaBotInferenceEngine:
    """Test suite for DattaBotInferenceEngine."""

    def test_initialization(self, inference_engine):
        """Test inference engine initializes correctly."""
        assert inference_engine.device.type == "cpu"
        assert inference_engine.vocab_size == 1000
        assert inference_engine.adaptive_softmax is not None
        assert inference_engine.agent_config.inference.max_new_tokens == 20

    def test_generate_response_structure(self, inference_engine):
        """Test that generated responses have correct DattaBotAPIResponse structure."""
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(1, 100, (batch_size, seq_len))

        responses = inference_engine.generate(input_ids, max_new_tokens=10)

        assert len(responses) == batch_size

        for response in responses:
            # Test DattaBotAPIResponse properties
            assert isinstance(response, DattaBotAPIResponse)

            # Test response dict properties
            assert isinstance(response.text, str)
            assert len(response.choices) > 0
            assert "text" in response.choices[0]
            assert "index" in response.choices[0]
            assert "finish_reason" in response.choices[0]

            # Test usage
            assert "prompt_tokens" in response.usage
            assert "completion_tokens" in response.usage
            assert "total_tokens" in response.usage
            assert (
                response.usage["total_tokens"]
                == response.usage["prompt_tokens"] + response.usage["completion_tokens"]
            )

            # Test metadata via properties
            assert response.tensor_response is not None
            assert isinstance(response.tokenizer_encodings, list)
            assert isinstance(response.tokenizer_decodings, list)
            assert len(response.tokenizer_encodings) == 1  # One list per batch item
            assert len(response.tokenizer_decodings) == 1

    def test_batch_generate(self, inference_engine):
        """Test batch text generation."""
        texts = ["Hello world", "How are you?", "What is AI?"]

        # Mock distributed module to prevent gather_object from being called
        with patch("src.inference_engine.dist.is_initialized", return_value=False):
            responses = inference_engine.batch_generate(texts, max_new_tokens=10)

        assert len(responses) == len(texts)
        for i, response in enumerate(responses):
            assert isinstance(response, DattaBotAPIResponse)
            assert response.text is not None
            assert len(response.tokenizer_encodings) == 1
            assert len(response.tokenizer_decodings) == 1

    def test_empty_batch_generate(self, inference_engine):
        """Test empty batch handling."""
        responses = inference_engine.batch_generate([])
        assert len(responses) == 0

    def test_generation_config_in_metadata(self, inference_engine):
        """Test that generation config is stored in metadata."""
        input_ids = torch.randint(1, 100, (1, 5))

        responses = inference_engine.generate(
            input_ids,
            max_new_tokens=15,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            do_sample=True,
            num_beams=2,
        )

        response = responses[0]
        gen_config = response._metadata["generation_config"]

        assert gen_config["max_new_tokens"] == 15
        assert gen_config["temperature"] == 0.8
        assert gen_config["top_k"] == 40
        assert gen_config["top_p"] == 0.95
        assert gen_config["do_sample"] is True
        assert gen_config["num_beams"] == 2

    def test_model_info_in_metadata(self, inference_engine):
        """Test that model info is stored in metadata."""
        input_ids = torch.randint(1, 100, (1, 5))
        responses = inference_engine.generate(input_ids, max_new_tokens=5)

        response = responses[0]
        model_info = response._metadata["model_info"]

        assert "d_model" in model_info
        assert "vocab_size" in model_info
        assert "device" in model_info
        assert model_info["vocab_size"] == 1000

    def test_finish_reason(self, inference_engine):
        """Test that finish_reason is correctly set."""
        input_ids = torch.randint(1, 100, (2, 5))

        # Generate with max tokens
        responses = inference_engine.generate(
            input_ids,
            max_new_tokens=inference_engine.max_response_tokens,
        )

        for response in responses:
            assert response.choices[0]["finish_reason"] in ["length", "stop"]

    def test_greedy_vs_sampling(self, inference_engine):
        """Test greedy vs sampling generation."""
        input_ids = torch.randint(1, 100, (1, 5))

        # Greedy
        responses_greedy = inference_engine.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
        )

        # Sampling
        responses_sample = inference_engine.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.8,
        )

        assert responses_greedy[0]._metadata["generation_config"]["do_sample"] is False
        assert responses_sample[0]._metadata["generation_config"]["do_sample"] is True

    def test_tensor_response_shape(self, inference_engine):
        """Test that tensor_response has correct shape."""
        batch_size = 3
        seq_len = 5
        input_ids = torch.randint(1, 100, (batch_size, seq_len))

        responses = inference_engine.generate(input_ids, max_new_tokens=10)

        for response in responses:
            tensor_resp = response.tensor_response
            assert tensor_resp.dim() == 1  # 1D tensor for single sequence
            assert tensor_resp.shape[0] >= seq_len  # At least input length

    def test_tokenizer_encodings_structure(self, inference_engine):
        """Test tokenizer encodings are list of lists."""
        input_ids = torch.randint(1, 100, (2, 5))
        responses = inference_engine.generate(input_ids, max_new_tokens=5)

        for response in responses:
            encodings = response.tokenizer_encodings
            assert isinstance(encodings, list)
            assert len(encodings) == 1  # One encoding list per response
            assert isinstance(encodings[0], list)
            assert all(isinstance(token_id, int) for token_id in encodings[0])

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, inference_engine, batch_size):
        """Test generation with different batch sizes."""
        input_ids = torch.randint(1, 100, (batch_size, 5))
        responses = inference_engine.generate(input_ids, max_new_tokens=10)

        assert len(responses) == batch_size
        for response in responses:
            assert isinstance(response, DattaBotAPIResponse)

    def test_usage_tokens_calculation(self, inference_engine):
        """Test that token counts in usage are accurate."""
        seq_len = 5
        max_new = 10
        input_ids = torch.randint(1, 100, (1, seq_len))

        responses = inference_engine.generate(input_ids, max_new_tokens=max_new)

        response = responses[0]
        usage = response.usage

        assert usage["prompt_tokens"] == seq_len
        assert usage["completion_tokens"] <= max_new
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

    def test_inference_time_tracking(self, inference_engine):
        """Test that inference time is tracked."""
        input_ids = torch.randint(1, 100, (2, 5))
        responses = inference_engine.generate(input_ids, max_new_tokens=5)

        for response in responses:
            assert "inference_time" in response._metadata
            assert response._metadata["inference_time"] > 0
            assert "tokens_per_second" in response._metadata
            assert response._metadata["tokens_per_second"] >= 0
