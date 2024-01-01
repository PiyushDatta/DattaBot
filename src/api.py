from traceback import print_exception
from src.agent import Agent as DattaBotAgent
from src.logger import get_logger


class DattaBotAPI:
    def __init__(self) -> None:
        self.agent = DattaBotAgent()

    def respond_to_queries(self, queries: list[str]) -> str:
        response = ""
        try:
            response = self.agent.respond_to_queries(queries=queries)
        except Exception as err:
            _logger = get_logger()
            _logger.error(f"Traceback: {print_exception(err)}")
            _logger.error(f"Error: {err}")
            response = "Got an error. Sorry! Please try again."
        return response
