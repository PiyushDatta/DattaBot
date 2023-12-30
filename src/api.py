from src.agent import Agent as DattaBotAgent


class DattaBotAPI:
    def __init__(self) -> None:
        self.agent = DattaBotAgent()

    def respond_to_query(self, input: str):
        return self.agent.respond_to_query(query=input)
