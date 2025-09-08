import pytest
import time
from unittest.mock import Mock

from src.conversation_manager import ConversationManager
from src.api_interface import DattaBotAPIResponse


@pytest.fixture
def mock_agent():
    """Fixture that returns a mocked Agent with tokenizer and responder."""
    agent = Mock()
    agent.respond_to_queries = Mock(
        return_value=[DattaBotAPIResponse({"output_text": "Hello from Agent"})]
    )
    agent.tokenizer_encode = Mock(return_value=[101, 202, 303])
    return agent


@pytest.fixture
def manager(mock_agent):
    return ConversationManager(agent=mock_agent)


def test_add_and_get_messages(manager):
    manager.add_message("user", "Hi there")
    msgs = manager.get_messages()

    assert len(msgs) == 1
    msg = msgs[0]
    assert msg["role"] == "user"
    assert msg["content"][0]["text"] == "Hi there"
    assert "id" in msg
    assert "created" in msg


def test_generate_reply_success(manager):
    manager.add_message("user", "How are you?")
    reply = manager.generate_reply()

    assert reply["role"] == "assistant"
    assert reply["content"][0]["text"] == "Hello from Agent"
    assert manager.get_messages()[-1] == reply
    manager.agent.respond_to_queries.assert_called_once_with(["How are you?"])


def test_generate_reply_raises_if_no_user_message(manager):
    with pytest.raises(ValueError):
        manager.generate_reply()

    manager.add_message("assistant", "I am fine")
    with pytest.raises(ValueError):
        manager.generate_reply()


def test_to_training_example(manager):
    manager.add_message("user", "Hi")
    manager.add_message("assistant", "Hello")
    result = manager.to_training_example()

    assert "text" in result
    assert "tokens" in result
    assert result["text"].startswith("USER: Hi")
    assert "ASSISTANT: Hello" in result["text"]
    manager.agent.tokenizer_encode.assert_called_once()


def test_message_timestamp_is_recent(manager):
    now = int(time.time())
    manager.add_message("user", "Check timestamp")
    msg = manager.get_messages()[0]
    assert abs(msg["created"] - now) < 3  # within a few seconds
