from __future__ import annotations
from traceback import print_exception
from typing import List, Optional
from src.api_interface import AgentAction, DattaBotAPIException, DattaBotAPIResponse
from src.logger import get_logger
from src.data_loader import DattabotDataBuilder


class DattaBotAPI:
    """
    DattaBot API wrapper that combines:
      - OpenAI Harmony responses
      - Local Agent actions (encoding, decoding, tensors, training)
    """

    def __init__(self, client: Optional["OpenAI"] = None):
        self.logger = get_logger()
        self._agent = None
        self._client = client

    def _ensure_agent(self):
        """Lazy-load the DattaBotAgent only when needed."""
        if self._agent is None:
            self.logger.debug("Initializing DattaBotAgent lazily...")
            # Import here instead of top-level to delay heavy torch load
            from src.agent import Agent as DattaBotAgent

            self._agent = DattaBotAgent()

    # --- Agent-based methods ---
    def respond_to_queries(self, queries: list[str]) -> list[DattaBotAPIResponse]:
        self._ensure_agent()
        return self._get_agent_action(
            queries=queries, action_type=AgentAction.GET_RESPONSES_FOR_QUERIES
        )

    def get_response(self, query: str) -> DattaBotAPIResponse:
        """Convenience method for a single Harmony query."""
        resp_list = self.respond_to_queries([query])
        return resp_list[0] if resp_list else DattaBotAPIResponse(None)

    def get_encoding(self, queries: List[str]) -> list[DattaBotAPIResponse]:
        self._ensure_agent()
        return self._get_agent_action(queries, AgentAction.GET_ENCODINGS_FOR_QUERIES)

    def get_decoding(self, queries: List[str]) -> list[DattaBotAPIResponse]:
        self._ensure_agent()
        return self._get_agent_action(queries, AgentAction.GET_DECODINGS_FOR_QUERIES)

    def get_tensor_encoding(self, queries: List[str]) -> list[DattaBotAPIResponse]:
        self._ensure_agent()
        return self._get_agent_action(
            queries, AgentAction.GET_ENCODED_TENSORS_FOR_QUERIES
        )

    def train_agent(self) -> list[DattaBotAPIResponse]:
        self._ensure_agent()
        return self._get_agent_action([], AgentAction.TRAIN_AGENT)

    def get_random_validation_example(self) -> DattaBotAPIResponse:
        """
        Returns a random example from the validation/test dataset.
        Useful for manual inspection or inference testing through the API.
        """
        response = DattaBotAPIResponse()
        try:
            builder = DattabotDataBuilder()
            response_dict: dict = None
            metadata: dict = None
            response_dict, metadata = builder.get_random_example()
            response = DattaBotAPIResponse(
                response_dict=response_dict, metadata=metadata
            )
        except Exception as err:
            self.logger.error(f"Failed to get random validation example: {err}")
            response.query_response = "Error fetching validation example."

        return response

    # --- Internal helper ---
    def _get_agent_action(
        self, queries: List[str], action_type: AgentAction
    ) -> list[DattaBotAPIResponse]:
        responses: list[DattaBotAPIResponse] = [DattaBotAPIResponse()]
        try:
            match action_type:
                case AgentAction.GET_RESPONSES_FOR_QUERIES:
                    responses = self._agent.respond_to_queries(queries=queries)
                case AgentAction.GET_ENCODINGS_FOR_QUERIES:
                    responses[0].tokenizer_encodings = self._agent.tokenizer_obj.encode(
                        text_or_texts=queries
                    )
                case AgentAction.GET_DECODINGS_FOR_QUERIES:
                    responses[0].tokenizer_decodings = self._agent.tokenizer_obj.decode(
                        tokens_or_tokens_list=queries
                    )
                case AgentAction.GET_ENCODED_TENSORS_FOR_QUERIES:
                    (
                        responses[0].tensor_response,
                        responses[0]._num_train_batches,
                    ) = self._agent.convert_queries_to_tensors(queries=queries)
                case AgentAction.TRAIN_AGENT:
                    responses[0] = self._agent.train_agent()
                case _:
                    raise DattaBotAPIException("Incorrect agent action requested.")
        except Exception as err:
            self.logger.error(f"Traceback: {print_exception(err)}")
            self.logger.error(f"Error: {err}")
            self.logger.error(f"Action type: {action_type}")
            responses[0].query_response = "Got an error. Sorry! Please try again."

        return responses
