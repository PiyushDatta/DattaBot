import pytest
from src.communication_mgr import DattaBotCommunicationManager
from src.api_interface import DattaBotAPIResponse


def test_add_user_and_agent_messages():
    cm = DattaBotCommunicationManager()

    cm.add_user_message("Hello, bot!")
    cm.add_agent_message("Hello, human!")

    history = cm.get_history()

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello, bot!"
    assert history[1]["role"] == "agent"
    assert history[1]["content"] == "Hello, human!"


def test_build_reply_adds_to_history():
    cm = DattaBotCommunicationManager()

    reply_text = "This is a reply."
    resp = cm.build_reply(reply_text)

    # Check return type
    assert isinstance(resp, DattaBotAPIResponse)
    assert resp.text == reply_text

    # Check that it was also stored in history
    history = cm.get_history()
    assert history[-1]["role"] == "agent"
    assert history[-1]["content"] == reply_text


def test_history_initially_empty():
    cm = DattaBotCommunicationManager()
    assert cm.get_history() == []
