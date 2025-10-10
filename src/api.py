from __future__ import annotations

from traceback import print_exception
from typing import List, Optional

from src.api_interface import DattaBotAPIException, DattaBotAPIResponse
from src.data_loader import DattabotDataBuilder
from src.evals.eval_engine import EvaluationEngine
from src.logger import get_logger
from src.util import AgentAction, is_rank_0


class DattaBotAPI:
    """
    DattaBot API wrapper that combines:
      - OpenAI Harmony responses
      - Local Agent actions (encoding, decoding, tensors, training)
      - Benchmark evaluations
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
        return self._get_agent_action(
            queries=queries, action_type=AgentAction.GET_RESPONSES_FOR_QUERIES
        )

    def get_response(self, query: str) -> DattaBotAPIResponse:
        """Convenience method for a single Harmony query."""
        resp_list = self.respond_to_queries([query])
        return resp_list[0] if resp_list else DattaBotAPIResponse(None)

    def get_encoding(self, queries: List[str]) -> list[DattaBotAPIResponse]:
        return self._get_agent_action(queries, AgentAction.GET_ENCODINGS_FOR_QUERIES)

    def get_decoding(self, queries: List[str]) -> list[DattaBotAPIResponse]:
        return self._get_agent_action(queries, AgentAction.GET_DECODINGS_FOR_QUERIES)

    def get_tensor_encoding(self, queries: List[str]) -> list[DattaBotAPIResponse]:
        return self._get_agent_action(
            queries, AgentAction.GET_ENCODED_TENSORS_FOR_QUERIES
        )

    def train_agent(self) -> list[DattaBotAPIResponse]:
        return self._get_agent_action([], AgentAction.TRAIN_AGENT)

    def get_random_validation_example(self) -> DattaBotAPIResponse:
        """
        Returns a random example from the validation/test dataset.
        Useful for manual inspection or inference testing through the API.
        """
        response = DattaBotAPIResponse()
        try:
            builder = DattabotDataBuilder()
            response_dict: Optional[dict] = None
            metadata: Optional[dict] = None
            response_dict, metadata = builder.get_random_example()
            response = DattaBotAPIResponse(
                response_dict=response_dict, metadata=metadata
            )
        except Exception as err:
            self.logger.error(f"Failed to get random validation example: {err}")
            response.text = "Error fetching validation example."

        return response

    def run_evaluation(self, queries: list[str]) -> list[DattaBotAPIResponse]:
        return self._get_agent_action(queries, AgentAction.RUN_EVALUATION)

    # --- Internal helper ---
    def _get_agent_action(
        self, queries: List[str], action_type: AgentAction
    ) -> list[DattaBotAPIResponse]:
        responses: list[DattaBotAPIResponse] = [DattaBotAPIResponse()]
        try:
            match action_type:
                case AgentAction.GET_RESPONSES_FOR_QUERIES:
                    self._ensure_agent()
                    assert self._agent is not None, "Agent is not set up."
                    responses = self._agent.respond_to_queries(queries=queries)
                case AgentAction.GET_ENCODINGS_FOR_QUERIES:
                    self._ensure_agent()
                    assert self._agent is not None, "Agent is not set up."
                    responses[0].tokenizer_encodings = self._agent.tokenizer_obj.encode(
                        text_or_texts=queries
                    )
                case AgentAction.GET_DECODINGS_FOR_QUERIES:
                    self._ensure_agent()
                    assert self._agent is not None, "Agent is not set up."
                    responses[0].tokenizer_decodings = self._agent.tokenizer_obj.decode(
                        tokens_or_tokens_list=queries
                    )
                case AgentAction.GET_ENCODED_TENSORS_FOR_QUERIES:
                    self._ensure_agent()
                    assert self._agent is not None, "Agent is not set up."
                    (
                        responses[0].tensor_response,
                        responses[0].num_train_batches,
                    ) = self._agent.convert_queries_to_tensors(queries=queries)
                case AgentAction.TRAIN_AGENT:
                    self._ensure_agent()
                    assert self._agent is not None, "Agent is not set up."
                    responses[0] = self._agent.train_agent()
                case AgentAction.RUN_EVALUATION:
                    if len(queries) > 0 and is_rank_0():
                        self._ensure_agent()
                        assert self._agent is not None, "Agent is not set up."
                        # Expected format: "benchmark,batch_size,num_samples,output_file"
                        benchmark = queries[0] if len(queries) > 0 else "humaneval"
                        batch_size = int(queries[1]) if len(queries) > 1 else 4
                        num_samples = int(queries[2]) if len(queries) > 2 else 1
                        output_file = (
                            queries[3]
                            if len(queries) > 3 and queries[3] != "None"
                            else None
                        )
                        responses[0] = EvaluationEngine.run_eval(
                            agent=self._agent,
                            benchmark=benchmark,
                            batch_size=batch_size,
                            num_samples=num_samples,
                            output_file=output_file,
                        )
                    elif not is_rank_0():
                        self.logger.warn(
                            f"Not rank 0 gpu, not running eval. Exiting now."
                        )
                    else:
                        raise DattaBotAPIException("No evaluation parameters provided")
                case _:
                    raise DattaBotAPIException("Incorrect agent action requested.")
        except Exception as err:
            self.logger.error(f"Traceback: {print_exception(err)}")
            self.logger.error(f"Error: {err}")
            self.logger.error(f"Action type: {action_type}")
            responses[0].text = "Got an error. Sorry! Please try again."

        return responses
