import pytest
from torch import Tensor, ones
from src.api_interface import AgentAction, DattaBotAPIException, DattaBotAPIResponse


# -----------------------------
# Test AgentAction enum
# -----------------------------
def test_agent_action_enum():
    assert AgentAction.NO_ACTION_START.value == 0
    assert AgentAction.GET_RESPONSES_FOR_QUERIES.value == 1
    assert AgentAction.NO_ACTION_END.value == 6
    assert int(AgentAction.TRAIN_AGENT) == 5
    assert list(AgentAction) == [
        AgentAction.NO_ACTION_START,
        AgentAction.GET_RESPONSES_FOR_QUERIES,
        AgentAction.GET_ENCODINGS_FOR_QUERIES,
        AgentAction.GET_DECODINGS_FOR_QUERIES,
        AgentAction.GET_ENCODED_TENSORS_FOR_QUERIES,
        AgentAction.TRAIN_AGENT,
        AgentAction.NO_ACTION_END
    ]


# -----------------------------
# Test DattaBotAPIException
# -----------------------------
def test_datta_bot_api_exception():
    with pytest.raises(DattaBotAPIException):
        raise DattaBotAPIException("Test error")


# -----------------------------
# Test DattaBotAPIResponse defaults and setters
# -----------------------------
def test_datta_bot_api_response_defaults_and_setters():
    resp = DattaBotAPIResponse()

    # Default Harmony response properties
    assert resp.text == ""
    assert resp.choices == []
    assert resp.usage == {}
    assert resp.raw == {}

    # Default metadata properties
    assert resp.tensor_response is None
    assert resp.num_train_batches == 0
    assert resp.num_val_batches == 0
    assert resp.tokenizer_encodings == []
    assert resp.tokenizer_decodings == []

    # Setters for Harmony response
    resp.text = "Hello"
    resp.choices = [{"text": "Option 1"}]
    resp.usage = {"tokens": 10}

    assert resp.text == "Hello"
    assert resp.choices == [{"text": "Option 1"}]
    assert resp.usage == {"tokens": 10}

    # Setters for metadata
    tensor = ones(2, 2)
    resp.tensor_response = tensor
    resp.num_train_batches = 5
    resp.num_val_batches = 3
    resp.tokenizer_encodings = [[1, 2, 3], [4, 5]]
    resp.tokenizer_decodings = ["hello", "world"]

    assert isinstance(resp.tensor_response, Tensor)
    assert resp.tensor_response.shape == (2, 2)
    assert resp.num_train_batches == 5
    assert resp.num_val_batches == 3
    assert resp.tokenizer_encodings == [[1, 2, 3], [4, 5]]
    assert resp.tokenizer_decodings == ["hello", "world"]

    # Reset metadata by setting None
    resp.num_train_batches = None
    resp.num_val_batches = None
    resp.tokenizer_encodings = None
    resp.tokenizer_decodings = None

    assert resp.num_train_batches == 0
    assert resp.num_val_batches == 0
    assert resp.tokenizer_encodings == []
    assert resp.tokenizer_decodings == []

    # String and repr
    s = str(resp)
    r = repr(resp)
    assert "Query Response" in s
    assert "DattaBotAPIResponse" in r
    assert "text=" in r


# -----------------------------
# Test initialization with pre-filled data
# -----------------------------
def test_datta_bot_api_response_init_with_data():
    response_dict = {
        "output_text": "Pre-filled text",
        "choices": [{"text": "Choice A"}],
        "usage": {"tokens": 42}
    }
    metadata = {
        "tensor_response": ones(1, 1),
        "num_train_batches": 2,
        "num_val_batches": 1,
        "tokenizer_encodings": [[1, 2]],
        "tokenizer_decodings": ["pre", "filled"]
    }

    resp = DattaBotAPIResponse(response_dict=response_dict, metadata=metadata)

    # Check Harmony response
    assert resp.text == "Pre-filled text"
    assert resp.choices == [{"text": "Choice A"}]
    assert resp.usage == {"tokens": 42}
    assert resp.raw == response_dict

    # Check metadata
    assert isinstance(resp.tensor_response, Tensor)
    assert resp.tensor_response.shape == (1, 1)
    assert resp.num_train_batches == 2
    assert resp.num_val_batches == 1
    assert resp.tokenizer_encodings == [[1, 2]]
    assert resp.tokenizer_decodings == ["pre", "filled"]

    # Update some values and check
    resp.text = "Updated"
    resp.num_train_batches = 10
    resp.num_val_batches = 7
    assert resp.text == "Updated"
    assert resp.num_train_batches == 10
    assert resp.num_val_batches == 7
