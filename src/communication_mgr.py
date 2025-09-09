# src/communication_manager.py
from typing import List, Dict, Any
from src.api_interface import DattaBotAPIResponse


class DattaBotCommunicationManager:
    """
    Manages conversation history and context for the Agent.
    This is internal and not exposed outside the Agent.
    """

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_agent_message(self, text: str) -> None:
        self.messages.append({"role": "agent", "content": text})

    def get_history(self) -> List[Dict[str, Any]]:
        return self.messages

    def build_reply(self, reply_text: str) -> DattaBotAPIResponse:
        """Wrap reply into DattaBotAPIResponse and add to history."""
        self.add_agent_message(reply_text)
        return DattaBotAPIResponse({"output_text": reply_text})
