import time
import uuid
from typing import List, Dict, Any

from src.api_interface import DattaBotAPIResponse


class ConversationManager:
    """
    Manages conversation state for the Agent using OpenAI Harmony format.
    """

    def __init__(self, agent: "Agent"):
        self.agent = agent
        self.messages: List[Dict[str, Any]] = []

    def _new_message(self, role: str, text: str) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "type": "message",
            "role": role,  # "user", "assistant", or "system"
            "content": [{"type": "text", "text": text}],
            "created": int(time.time()),
        }

    def add_message(self, role: str, text: str) -> None:
        """Add a message (user or assistant)."""
        self.messages.append(self._new_message(role, text))

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return full conversation history in Harmony format."""
        return self.messages

    def generate_reply(self) -> Dict[str, Any]:
        """
        Generate a reply from the Agent based on conversation history.
        """
        if not self.messages or self.messages[-1]["role"] != "user":
            raise ValueError("Last message must be from user before generating reply.")

        user_text = self.messages[-1]["content"][0]["text"]
        responses: list[DattaBotAPIResponse] = self.agent.respond_to_queries(
            [user_text]
        )
        # Assert all items are DattaBotAPIResponse
        assert isinstance(responses, list), f"Expected list, got {type(responses)}"
        assert all(
            isinstance(r, DattaBotAPIResponse) for r in responses
        ), "Not all responses are DattaBotAPIResponse"
        reply_text = responses[0].text
        reply_msg = self._new_message("assistant", reply_text)
        self.messages.append(reply_msg)
        return reply_msg

    def to_training_example(self) -> Dict[str, Any]:
        """
        Convert conversation into a tokenized training example.
        """
        dialogue_str = ""
        for msg in self.messages:
            prefix = f"{msg['role'].upper()}: "
            dialogue_str += prefix + msg["content"][0]["text"] + "\n"

        encoded = self.agent.tokenizer_encode([dialogue_str])
        return {
            "text": dialogue_str,
            "tokens": encoded,
        }
