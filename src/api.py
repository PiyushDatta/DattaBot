from enum import IntEnum
from traceback import print_exception
from src.agent import Agent as DattaBotAgent
from src.logger import get_logger
from src.api_interface import AgentAction, DattaBotAPIResponse, DattaBotAPIException


class DattaBotAPI:
    def __init__(self) -> None:
        self.agent = DattaBotAgent()

    def respond_to_queries(self, queries: list[str]) -> DattaBotAPIResponse:
        return self._get_agent_action(
            queries=queries, action_type=AgentAction.GET_RESPONSES_FOR_QUERIES
        )

    def get_encoding(self, queries: list[str]) -> DattaBotAPIResponse:
        return self._get_agent_action(
            queries=queries, action_type=AgentAction.GET_ENCODINGS_FOR_QUERIES
        )

    def get_decoding(self, queries: list[str]) -> DattaBotAPIResponse:
        return self._get_agent_action(
            queries=queries, action_type=AgentAction.GET_DECODINGS_FOR_QUERIES
        )

    def get_tensor_encoding(self, queries: list[str]) -> DattaBotAPIResponse:
        return self._get_agent_action(
            queries=queries, action_type=AgentAction.GET_ENCODED_TENSORS_FOR_QUERIES
        )

    def train_agent(self) -> DattaBotAPIResponse:
        return self._get_agent_action(queries=[], action_type=AgentAction.TRAIN_AGENT)

    def _get_agent_action(
        self, queries: list[str], action_type: AgentAction
    ) -> DattaBotAPIResponse:
        response: DattaBotAPIResponse = DattaBotAPIResponse()
        try:
            match action_type:
                case AgentAction.GET_RESPONSES_FOR_QUERIES:
                    response = self.agent.respond_to_queries(queries=queries)
                case AgentAction.GET_ENCODINGS_FOR_QUERIES:
                    response.tokenizer_encodings = self.agent.tokenizer_encode(
                        decoded_queries=queries
                    )
                case AgentAction.GET_DECODINGS_FOR_QUERIES:
                    response.tokenizer_decodings = self.agent.tokenizer_decode(
                        encoded_queries=queries
                    )
                case AgentAction.GET_ENCODED_TENSORS_FOR_QUERIES:
                    (
                        # Tensor
                        response.tensor_response,
                        # int
                        response.num_batches,
                    ) = self.agent.convert_queries_to_tensors(queries=queries)
                case AgentAction.TRAIN_AGENT:
                    response = self.agent.train_agent()
                case _:
                    raise DattaBotAPIException("Incorrect agent action requested.")
        except Exception as err:
            _logger = get_logger()
            _logger.error(f"Traceback: {print_exception(err)}")
            _logger.error(f"Error: {err}")
            _logger.error(f"Action type: {action_type}")
            response.query_response = "Got an error. Sorry! Please try again."

        return response
